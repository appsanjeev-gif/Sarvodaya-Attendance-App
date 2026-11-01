package com.school.sarvodayaattendance.utils

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Matrix
import android.graphics.Rect
import android.util.Base64
import com.google.mlkit.vision.face.Face
import com.google.mlkit.vision.face.FaceDetection
import com.google.mlkit.vision.face.FaceDetector
import com.google.mlkit.vision.face.FaceDetectorOptions
import com.google.mlkit.vision.face.FaceLandmark
import com.school.sarvodayaattendance.data.Student
import org.json.JSONArray
import org.json.JSONObject
import org.tensorflow.lite.Interpreter
import java.io.ByteArrayOutputStream
import java.io.File
import java.io.FileInputStream
import java.io.FileOutputStream
import java.net.HttpURLConnection
import java.net.URL
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.math.abs
import kotlin.math.atan2
import kotlin.math.max
import kotlin.math.sqrt

object FaceMatcherEngine {

    data class MatchResult(
        val student: Student?,
        val distance: Float,
        val cosineSimilarity: Float,
        val marginGap: Float,
        val isHighConfidence: Boolean,
        val topCandidates: List<Student>
    )

    private data class StudentBiometricTemplate(
        val student: Student,
        val poseEmbeddings: List<FloatArray>,
        val centroidEmbedding: FloatArray
    )

    @Volatile
    private var tfliteInterpreter: Interpreter? = null
    private var embeddingDim: Int = 192
    private var inputImageSize: Int = 112
    private val isDownloadingModel = AtomicBoolean(false)
    private val templateCache = ConcurrentHashMap<String, StudentBiometricTemplate>()

    // MobileFaceNet 192-D Cosine Similarity मानक:
    // एक ही व्यक्ति का Cosine Similarity >= 0.68 (यानी Distance <= 3.2f) होता है।
    // अलग व्यक्ति का Cosine Similarity हमेशा < 0.45 (यानी Distance > 5.5f) होता है।
    const val HIGH_CONFIDENCE_MAX_DISTANCE = 3.10f   // Cosine >= 0.69
    const val AI_VERIFY_MAX_DISTANCE = 4.30f         // Cosine >= 0.57 (केवल Gemini क्रॉस-चेक के लिए)
    const val MIN_MARGIN_GAP = 0.75f                 // 1st और 2nd मैच के बीच आवश्यक अंतर

    // विश्वसनीय ओपन-सोर्स MobileFaceNet (.tflite) मॉडल मिरर्स (Auto-Download के लिए)
    private val MODEL_DOWNLOAD_URLS = listOf(
        "https://raw.githubusercontent.com/sirius-ai/MobileFaceNet_TF/master/arch/pretrained_model/mobilefacenet.tflite",
        "https://github.com/MCarlomagno/FaceRecognitionAuth/raw/master/app/src/main/assets/mobile_face_net.tflite",
        "https://raw.githubusercontent.com/MCarlomagno/FaceRecognitionAuth/master/app/src/main/assets/mobile_face_net.tflite",
        "https://raw.githubusercontent.com/shubham0204/FaceRecognition_With_FaceNet_Android/master/app/src/main/assets/facenet.tflite"
    )

    val detector: FaceDetector by lazy {
        val options = FaceDetectorOptions.Builder()
            .setPerformanceMode(FaceDetectorOptions.PERFORMANCE_MODE_ACCURATE)
            .setLandmarkMode(FaceDetectorOptions.LANDMARK_MODE_ALL)
            .setClassificationMode(FaceDetectorOptions.CLASSIFICATION_MODE_ALL)
            .setMinFaceSize(0.20f)
            .build()
        FaceDetection.getClient(options)
    }

    fun isNeuralModelReady(): Boolean = (tfliteInterpreter != null)

    fun initEngine(context: Context) {
        if (tfliteInterpreter != null) return
        val appCtx = context.applicationContext

        // 1. पहले फोन के परमानेंट स्टोरेज में डाउनलोड किया हुआ मॉडल चेक करें
        val modelDir = File(appCtx.filesDir, "biometric_models").apply { if (!exists()) mkdirs() }
        val cachedModelFile = File(modelDir, "mobile_face_net.tflite")

        if (cachedModelFile.exists() && cachedModelFile.length() > 500_000L) {
            if (loadInterpreterFromFile(cachedModelFile)) return
        }

        // 2. यदि assets में मौजूद हो तो वहां से कॉपी करके लोड करें
        val assetNames = listOf("mobile_face_net.tflite", "mobilefacenet.tflite", "facenet.tflite")
        for (name in assetNames) {
            try {
                appCtx.assets.open(name).use { input ->
                    FileOutputStream(cachedModelFile).use { output ->
                        input.copyTo(output)
                    }
                }
                if (cachedModelFile.exists() && cachedModelFile.length() > 500_000L) {
                    if (loadInterpreterFromFile(cachedModelFile)) return
                }
            } catch (e: Exception) {}
        }

        // 3. यदि फोन में नहीं है, तो पहली बार खुलने पर खुद डाउनलोड करके हमेशा के लिए सेव कर ले
        downloadModelInBackground(cachedModelFile)
    }

    @Synchronized
    private fun loadInterpreterFromFile(file: File): Boolean {
        if (tfliteInterpreter != null) return true
        return try {
            FileInputStream(file).use { fis ->
                val mapped: MappedByteBuffer = fis.channel.map(
                    FileChannel.MapMode.READ_ONLY,
                    0,
                    file.length()
                )
                val opts = Interpreter.Options().apply { setNumThreads(4) }
                val interp = Interpreter(mapped, opts)
                val inShape = interp.getInputTensor(0).shape()
                if (inShape.size >= 3) {
                    inputImageSize = inShape[1]
                }
                val outShape = interp.getOutputTensor(0).shape()
                if (outShape.isNotEmpty()) {
                    embeddingDim = outShape.last()
                }
                tfliteInterpreter = interp
                templateCache.clear()
                true
            }
        } catch (e: Exception) {
            try { file.delete() } catch (ex: Exception) {}
            false
        }
    }

    private fun downloadModelInBackground(targetFile: File) {
        if (!isDownloadingModel.compareAndSet(false, true)) return
        Thread {
            try {
                for (urlStr in MODEL_DOWNLOAD_URLS) {
                    try {
                        val tempFile = File(targetFile.parentFile, "temp_model.tflite")
                        val conn = (URL(urlStr).openConnection() as HttpURLConnection).apply {
                            requestMethod = "GET"
                            connectTimeout = 10000
                            readTimeout = 20000
                            instanceFollowRedirects = true
                        }
                        if (conn.responseCode in 200..299) {
                            conn.inputStream.use { input ->
                                FileOutputStream(tempFile).use { output ->
                                    input.copyTo(output)
                                }
                            }
                            if (tempFile.exists() && tempFile.length() > 500_000L) {
                                if (targetFile.exists()) targetFile.delete()
                                tempFile.renameTo(targetFile)
                                if (loadInterpreterFromFile(targetFile)) {
                                    break
                                }
                            }
                        }
                    } catch (e: Exception) {}
                }
            } finally {
                isDownloadingModel.set(false)
            }
        }.start()
    }

    fun clearCache() {
        templateCache.clear()
    }

    fun warmUpCache(students: List<Student>) {
        Thread {
            for (stu in students) {
                if (stu.photos.isNotEmpty() && !templateCache.containsKey(stu.id)) {
                    val built = buildStudentTemplate(stu)
                    if (built != null) {
                        templateCache[stu.id] = built
                    }
                }
            }
        }.start()
    }

    fun decodeBase64Bitmap(base64Str: String): Bitmap? {
        return try {
            val clean = base64Str.substringAfter("base64,", base64Str)
            val bytes = Base64.decode(clean, Base64.DEFAULT)
            BitmapFactory.decodeByteArray(bytes, 0, bytes.size)
        } catch (e: Exception) {
            null
        }
    }

    // ✅ सख्त बायोमेट्रिक नियम: दोनों आँखें खुली होनी चाहिए और चेहरा सीधा होना चाहिए
    fun isLiveFaceValidForScan(face: Face): Boolean {
        if (abs(face.headEulerAngleY) > 16f || abs(face.headEulerAngleZ) > 15f || abs(face.headEulerAngleX) > 16f) {
            return false
        }
        val leftOpen = face.leftEyeOpenProbability ?: 1.0f
        val rightOpen = face.rightEyeOpenProbability ?: 1.0f
        if (leftOpen < 0.50f || rightOpen < 0.50f) {
            return false
        }
        val leftEye = face.getLandmark(FaceLandmark.LEFT_EYE)?.position
        val rightEye = face.getLandmark(FaceLandmark.RIGHT_EYE)?.position
        val nose = face.getLandmark(FaceLandmark.NOSE_BASE)?.position
        return leftEye != null && rightEye != null && nose != null
    }

    fun validateKycFaceQuality(face: Face, frameBmp: Bitmap): Pair<Boolean, String> {
        val box = face.boundingBox
        if (box.width() < 95 || box.height() < 95) {
            return Pair(false, "⚠️ कृपया कैमरा चेहरे के थोड़ा और पास लाएं")
        }
        if (abs(face.headEulerAngleY) > 15f || abs(face.headEulerAngleZ) > 14f) {
            return Pair(false, "⚠️ कृपया चेहरा बिल्कुल सीधा रखें (तिरछा न करें)")
        }
        val leftEyeOpen = face.leftEyeOpenProbability ?: 1.0f
        val rightEyeOpen = face.rightEyeOpenProbability ?: 1.0f
        if (leftEyeOpen < 0.50f || rightEyeOpen < 0.50f) {
            return Pair(false, "⚠️ कृपया दोनों आँखें पूरी तरह खुली रखें")
        }
        return Pair(true, "OK")
    }

    // ✅ Eye-Center Horizontal Alignment + Biometric 112x112 Crop
    fun alignAndCropFaceBitmap(source: Bitmap, face: Face): Bitmap {
        var workingBmp = source
        val box = face.boundingBox

        val leftEye = face.getLandmark(FaceLandmark.LEFT_EYE)?.position
        val rightEye = face.getLandmark(FaceLandmark.RIGHT_EYE)?.position

        if (leftEye != null && rightEye != null) {
            val dy = (rightEye.y - leftEye.y).toDouble()
            val dx = (rightEye.x - leftEye.x).toDouble()
            val angleDeg = Math.toDegrees(atan2(dy, dx)).toFloat()
            if (abs(angleDeg) in 1.5f..28.0f) {
                val cx = (leftEye.x + rightEye.x) / 2f
                val cy = (leftEye.y + rightEye.y) / 2f
                val matrix = Matrix().apply { postRotate(-angleDeg, cx, cy) }
                try {
                    workingBmp = Bitmap.createBitmap(source, 0, 0, source.width, source.height, matrix, true)
                } catch (e: Exception) {
                    workingBmp = source
                }
            }
        }
        return cropFaceBitmap(workingBmp, box)
    }

    fun cropFaceBitmap(source: Bitmap, boundingBox: Rect): Bitmap {
        val w = boundingBox.width()
        val h = boundingBox.height()
        val insetX = (w * 0.05f).toInt()
        val insetTop = (h * 0.06f).toInt()
        val insetBottom = (h * 0.03f).toInt()

        val left = (boundingBox.left + insetX).coerceIn(0, max(0, source.width - 1))
        val top = (boundingBox.top + insetTop).coerceIn(0, max(0, source.height - 1))
        val right = (boundingBox.right - insetX).coerceIn(left + 1, source.width)
        val bottom = (boundingBox.bottom - insetBottom).coerceIn(top + 1, source.height)

        val cropped = Bitmap.createBitmap(source, left, top, right - left, bottom - top)
        return Bitmap.createScaledBitmap(cropped, 160, 160, true)
    }

    private fun decodeStoredPhoto(base64Str: String): Bitmap? {
        return try {
            val rawBmp = decodeBase64Bitmap(base64Str) ?: return null
            val w = rawBmp.width
            val h = rawBmp.height
            if (w > 60 && h > 60) {
                val ix = (w * 0.05f).toInt()
                val iy = (h * 0.05f).toInt()
                val sub = Bitmap.createBitmap(rawBmp, ix, iy, w - ix * 2, h - iy * 2)
                Bitmap.createScaledBitmap(sub, 160, 160, true)
            } else {
                Bitmap.createScaledBitmap(rawBmp, 160, 160, true)
            }
        } catch (e: Exception) {
            null
        }
    }

    // ✅ असली 192-D MobileFaceNet (.tflite) बायोमेट्रिक एम्बेडिंग (कोई 2D पिक्सल मिलान नहीं)
    fun computeEmbedding(faceBmp: Bitmap): FloatArray? {
        val interp = tfliteInterpreter ?: return null
        return try {
            val scaled = Bitmap.createScaledBitmap(faceBmp, inputImageSize, inputImageSize, true)
            val byteBuffer = ByteBuffer.allocateDirect(1 * inputImageSize * inputImageSize * 3 * 4)
            byteBuffer.order(ByteOrder.nativeOrder())

            val pixels = IntArray(inputImageSize * inputImageSize)
            scaled.getPixels(pixels, 0, inputImageSize, 0, 0, inputImageSize, inputImageSize)

            // MobileFaceNet मानक प्री-प्रोसेसिंग: (pixel - 127.5f) / 128.0f
            for (px in pixels) {
                val r = ((px shr 16) and 0xFF).toFloat()
                val g = ((px shr 8) and 0xFF).toFloat()
                val b = (px and 0xFF).toFloat()
                byteBuffer.putFloat((r - 127.5f) / 128.0f)
                byteBuffer.putFloat((g - 127.5f) / 128.0f)
                byteBuffer.putFloat((b - 127.5f) / 128.0f)
            }

            val output = Array(1) { FloatArray(embeddingDim) }
            synchronized(interp) {
                interp.run(byteBuffer, output)
            }
            l2Normalize(output[0])
        } catch (e: Exception) {
            null
        }
    }

    private fun l2Normalize(vec: FloatArray): FloatArray {
        var sumSq = 0f
        for (v in vec) sumSq += v * v
        val norm = max(sqrt(sumSq), 1e-6f)
        val out = FloatArray(vec.size)
        for (i in vec.indices) out[i] = vec[i] / norm
        return out
    }

    private fun averageEmbeddings(list: List<FloatArray>): FloatArray {
        if (list.isEmpty()) return FloatArray(0)
        val len = list[0].size
        val avg = FloatArray(len)
        for (vec in list) {
            if (vec.size == len) {
                for (i in 0 until len) avg[i] += vec[i]
            }
        }
        return l2Normalize(avg)
    }

    private fun calculateCosineSimilarity(v1: FloatArray, v2: FloatArray): Float {
        if (v1.isEmpty() || v1.size != v2.size) return -1f
        var dot = 0f
        for (i in v1.indices) {
            dot += v1[i] * v2[i]
        }
        return dot.coerceIn(-1f, 1f)
    }

    private fun buildStudentTemplate(student: Student): StudentBiometricTemplate? {
        if (student.photos.isEmpty()) return null
        val vectors = mutableListOf<FloatArray>()
        for (b64 in student.photos) {
            val bmp = decodeStoredPhoto(b64)
            if (bmp != null) {
                val emb = computeEmbedding(bmp)
                if (emb != null) {
                    vectors.add(emb)
                }
            }
        }
        if (vectors.isEmpty()) return null
        val centroid = averageEmbeddings(vectors)
        return StudentBiometricTemplate(
            student = student,
            poseEmbeddings = vectors,
            centroidEmbedding = centroid
        )
    }

    fun findBestMatchDetailed(liveFaceBmp: Bitmap, candidates: List<Student>): MatchResult {
        if (candidates.isEmpty()) {
            return MatchResult(null, 999f, 0f, 0f, false, emptyList())
        }

        // यदि .tflite मॉडल अभी डाउनलोड हो रहा है (पहली बार), तो पहले 3 उम्मीदवारों को ऑनलाइन वेरिफिकेशन के लिए भेजें
        val liveVec = computeEmbedding(liveFaceBmp)
        if (liveVec == null) {
            val fallbackCandidates = candidates.filter { it.photos.isNotEmpty() }.take(3)
            return MatchResult(null, 999f, 0f, 0f, false, fallbackCandidates)
        }

        val scoredList = mutableListOf<Triple<Student, Float, Float>>() // Student, Distance, CosineSim

        for (stu in candidates) {
            if (stu.photos.isEmpty()) continue
            var template = templateCache[stu.id]
            if (template == null) {
                template = buildStudentTemplate(stu)
                if (template != null) {
                    templateCache[stu.id] = template
                }
            }
            if (template == null) continue

            var maxPoseCos = -1f
            for (poseVec in template.poseEmbeddings) {
                val cos = calculateCosineSimilarity(liveVec, poseVec)
                if (cos > maxPoseCos) maxPoseCos = cos
            }
            val centroidCos = calculateCosineSimilarity(liveVec, template.centroidEmbedding)
            val combinedCos = (maxPoseCos * 0.65f) + (centroidCos * 0.35f)
            val distance = (1f - combinedCos) * 10.0f // 0 = 100% समान, कम संख्या = बेहतर मैच

            scoredList.add(Triple(stu, distance, combinedCos))
        }

        if (scoredList.isEmpty()) {
            return MatchResult(null, 999f, 0f, 0f, false, emptyList())
        }

        scoredList.sortBy { it.second }

        val bestStudent = scoredList[0].first
        val bestDistance = scoredList[0].second
        val bestCosine = scoredList[0].third
        val secondBestDistance = if (scoredList.size > 1) scoredList[1].second else 10.0f
        val marginGap = secondBestDistance - bestDistance

        val topCandidates = scoredList
            .filter { it.second <= AI_VERIFY_MAX_DISTANCE }
            .take(3)
            .map { it.first }

        val isHighConf = (bestDistance <= HIGH_CONFIDENCE_MAX_DISTANCE) && (marginGap >= MIN_MARGIN_GAP)

        return if (isHighConf) {
            MatchResult(bestStudent, bestDistance, bestCosine, marginGap, true, topCandidates)
        } else if (topCandidates.isNotEmpty()) {
            MatchResult(bestStudent, bestDistance, bestCosine, marginGap, false, topCandidates)
        } else {
            MatchResult(null, bestDistance, bestCosine, marginGap, false, emptyList())
        }
    }

    fun findBestMatch(liveFaceBmp: Bitmap, candidates: List<Student>): Pair<Student?, Float> {
        val detailed = findBestMatchDetailed(liveFaceBmp, candidates)
        return Pair(detailed.student, detailed.distance)
    }

    // ✅ शुद्ध बायोमेट्रिक क्रॉस-चेक (बिना सिंदूर, कपड़े या किसी बाहरी चीज़ के—केवल चेहरे की हड्डियों और लिंग का मिलान)
    fun identifyStudentWithGeminiVisionBlocking(
        context: Context,
        liveFaceBmp: Bitmap,
        topCandidates: List<Student>
    ): Student? {
        if (topCandidates.isEmpty()) return null
        try {
            val vm = RealisticVoiceManager(context)
            val rawKeys = vm.getGeminiKeysString()
            val keyPool = rawKeys.split(",", "\n", ";", " ")
                .map { it.trim() }
                .filter { it.length > 15 && !it.startsWith("sk_", ignoreCase = true) }
            val apiKey = keyPool.firstOrNull() ?: return null

            val liveBaos = ByteArrayOutputStream()
            Bitmap.createScaledBitmap(liveFaceBmp, 200, 200, true)
                .compress(Bitmap.CompressFormat.JPEG, 85, liveBaos)
            val liveB64 = Base64.encodeToString(liveBaos.toByteArray(), Base64.NO_WRAP)

            val partsArray = JSONArray()
            val candidateDesc = StringBuilder()

            partsArray.put(JSONObject().put("text", "LIVE CAMERA FACE (Image 0):"))
            partsArray.put(JSONObject().put("inlineData", JSONObject().apply {
                put("mimeType", "image/jpeg")
                put("data", liveB64)
            }))

            val validList = mutableListOf<Student>()
            for ((idx, stu) in topCandidates.withIndex()) {
                if (stu.photos.isEmpty()) continue
                val storedClean = stu.photos.first().substringAfter("base64,")
                val num = idx + 1
                validList.add(stu)
                candidateDesc.append("Candidate $num: Name='${stu.name}', RegisteredGender='${stu.gender}'\n")
                partsArray.put(JSONObject().put("text", "CANDIDATE $num REFERENCE PHOTO:"))
                partsArray.put(JSONObject().put("inlineData", JSONObject().apply {
                    put("mimeType", "image/jpeg")
                    put("data", storedClean)
                }))
            }

            if (validList.isEmpty()) return null

            val strictPrompt = """
                You are a pure biometric facial geometry verification engine.
                Compare the LIVE CAMERA FACE (Image 0) against the ${validList.size} candidate reference photos:
                $candidateDesc
                
                BIOMETRIC RULES (IGNORE CLOTHING, ACCESSORIES, OR MAKEUP):
                1. Rely ONLY on permanent craniofacial bone structure: interpupillary distance, eye socket shape, nasal bridge width, philtrum, jawline, and biological age group.
                2. If the LIVE CAMERA FACE is an adult man or adult woman and does not match the child's craniofacial structure in the reference photo, reply "NONE".
                3. Biological gender of the live face must match the candidate photo.
                4. Only if you are 99.8% certain that Image 0 is the exact same person as one of the candidates, reply with the candidate number ("1", "2", or "3"). Otherwise reply "NONE".
            """.trimIndent()

            partsArray.put(JSONObject().put("text", strictPrompt))

            val endpoint = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key=$apiKey"
            val conn = (URL(endpoint).openConnection() as HttpURLConnection).apply {
                requestMethod = "POST"
                connectTimeout = 5000
                readTimeout = 6500
                doOutput = true
                setRequestProperty("Content-Type", "application/json; charset=UTF-8")
            }

            val payload = JSONObject().apply {
                put("contents", JSONArray().put(JSONObject().put("parts", partsArray)))
            }

            conn.outputStream.use { it.write(payload.toString().toByteArray(Charsets.UTF_8)) }
            if (conn.responseCode == 200) {
                val resp = conn.inputStream.bufferedReader(Charsets.UTF_8).use { it.readText() }
                val rawAns = JSONObject(resp)
                    .optJSONArray("candidates")?.optJSONObject(0)
                    ?.optJSONObject("content")?.optJSONArray("parts")?.optJSONObject(0)
                    ?.optString("text", "")?.trim()?.uppercase() ?: "NONE"

                val chosenIdx = when {
                    rawAns == "1" || rawAns.startsWith("1") -> 0
                    rawAns == "2" || rawAns.startsWith("2") -> 1
                    rawAns == "3" || rawAns.startsWith("3") -> 2
                    else -> -1
                }
                if (chosenIdx in validList.indices) {
                    return validList[chosenIdx]
                }
            }
        } catch (e: Exception) {}
        return null
    }
}
