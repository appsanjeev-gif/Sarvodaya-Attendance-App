package com.school.sarvodayaattendance.utils

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.graphics.ColorMatrix
import android.graphics.ColorMatrixColorFilter
import android.graphics.Matrix
import android.graphics.Paint
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
import kotlin.math.abs
import kotlin.math.atan2
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt
import kotlin.math.sqrt

object FaceMatcherEngine {

    data class MatchResult(
        val student: Student?,
        val distance: Float,
        val matchPercentage: Int,
        val cosineSimilarity: Float,
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
    private val templateCache = ConcurrentHashMap<String, StudentBiometricTemplate>()

    // संतुलित बायोमेट्रिक थ्रेशोल्ड (Balanced MobileFaceNet Cosine Similarity):
    // Cosine Similarity >= 0.58 (यानी Match >= 68%) होने पर High Confidence पास होगा।
    const val PASS_COSINE_THRESHOLD = 0.58f
    const val AI_ASSIST_COSINE_MIN = 0.46f
    const val MIN_COSINE_MARGIN_GAP = 0.045f

    val detector: FaceDetector by lazy {
        val options = FaceDetectorOptions.Builder()
            .setPerformanceMode(FaceDetectorOptions.PERFORMANCE_MODE_ACCURATE)
            .setLandmarkMode(FaceDetectorOptions.LANDMARK_MODE_ALL)
            .setClassificationMode(FaceDetectorOptions.CLASSIFICATION_MODE_ALL)
            .setMinFaceSize(0.18f)
            .build()
        FaceDetection.getClient(options)
    }

    fun isNeuralModelReady(): Boolean = (tfliteInterpreter != null)

    @Synchronized
    fun initEngine(context: Context) {
        if (tfliteInterpreter != null) return
        val appCtx = context.applicationContext

        val assetNames = listOf("mobile_face_net.tflite", "mobilefacenet.tflite", "facenet.tflite")
        for (name in assetNames) {
            try {
                val afd = appCtx.assets.openFd(name)
                FileInputStream(afd.fileDescriptor).use { fis ->
                    val mapped: MappedByteBuffer = fis.channel.map(
                        FileChannel.MapMode.READ_ONLY,
                        afd.startOffset,
                        afd.declaredLength
                    )
                    if (setupInterpreter(mapped)) return
                }
            } catch (e: Exception) {
                try {
                    val cacheFile = File(appCtx.filesDir, "packed_$name")
                    appCtx.assets.open(name).use { input ->
                        FileOutputStream(cacheFile).use { output ->
                            input.copyTo(output)
                        }
                    }
                    if (cacheFile.exists() && cacheFile.length() > 500_000L) {
                        FileInputStream(cacheFile).use { fis ->
                            val mapped = fis.channel.map(FileChannel.MapMode.READ_ONLY, 0, cacheFile.length())
                            if (setupInterpreter(mapped)) return
                        }
                    }
                } catch (ex: Exception) {}
            }
        }
    }

    private fun setupInterpreter(mapped: MappedByteBuffer): Boolean {
        return try {
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
        } catch (e: Exception) {
            false
        }
    }

    fun clearCache() {
        templateCache.clear()
    }

    fun warmUpCache(students: List<Student>) {
        Thread {
            for (stu in students) {
                if ((stu.embeddings.isNotEmpty() || stu.photos.isNotEmpty()) && !templateCache.containsKey(stu.id)) {
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

    fun isLiveFaceValidForScan(face: Face): Boolean {
        if (abs(face.headEulerAngleY) > 20f || abs(face.headEulerAngleZ) > 18f) {
            return false
        }
        val leftOpen = face.leftEyeOpenProbability ?: 1.0f
        val rightOpen = face.rightEyeOpenProbability ?: 1.0f
        if (leftOpen < 0.42f || rightOpen < 0.42f) {
            return false
        }
        return true
    }

    fun validateKycFaceQuality(face: Face, frameBmp: Bitmap): Pair<Boolean, String> {
        val box = face.boundingBox
        if (box.width() < 85 || box.height() < 85) {
            return Pair(false, "⚠️ कृपया कैमरा चेहरे के थोड़ा और पास लाएं")
        }
        if (abs(face.headEulerAngleY) > 18f || abs(face.headEulerAngleZ) > 16f) {
            return Pair(false, "⚠️ कृपया चेहरा बिल्कुल सीधा रखें")
        }
        val leftEyeOpen = face.leftEyeOpenProbability ?: 1.0f
        val rightEyeOpen = face.rightEyeOpenProbability ?: 1.0f
        if (leftEyeOpen < 0.40f || rightEyeOpen < 0.40f) {
            return Pair(false, "⚠️ कृपया दोनों आँखें खुली रखें")
        }
        return Pair(true, "OK")
    }

    // ✅ 1. DP (प्रोफाइल फोटो) के लिए सुंदर पूरा चेहरा (सिर के बालों से कंधे तक + AI Studio Portrait Enhancement)
    fun createAiEnhancedPortraitDp(source: Bitmap, face: Face): Bitmap {
        val box = face.boundingBox
        val w = box.width()
        val h = box.height()

        // पूरा सिर, बाल, कान और कंधे का थोड़ा हिस्सा शामिल करें (Full Natural Portrait DP)
        val padX = (w * 0.32f).toInt()
        val padTop = (h * 0.40f).toInt()
        val padBottom = (h * 0.35f).toInt()

        val left = (box.left - padX).coerceIn(0, max(0, source.width - 1))
        val top = (box.top - padTop).coerceIn(0, max(0, source.height - 1))
        val right = (box.right + padX).coerceIn(left + 1, source.width)
        val bottom = (box.bottom + padBottom).coerceIn(top + 1, source.height)

        val portraitCrop = Bitmap.createBitmap(source, left, top, right - left, bottom - top)
        val scaledDp = Bitmap.createScaledBitmap(portraitCrop, 300, 300, true)

        // AI Portrait Lighting & Clarity Enhancement (Auto Contrast + Natural Studio Warmth)
        val enhanced = Bitmap.createBitmap(300, 300, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(enhanced)
        val cm = ColorMatrix(
            floatArrayOf(
                1.08f, 0f, 0f, 0f, 6f,
                0f, 1.07f, 0f, 0f, 5f,
                0f, 0f, 1.05f, 0f, 3f,
                0f, 0f, 0f, 1f, 0f
            )
        )
        val paint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            colorFilter = ColorMatrixColorFilter(cm)
        }
        canvas.drawBitmap(scaledDp, 0f, 0f, paint)
        return enhanced
    }

    // ✅ 2. बायोमेट्रिक 192-D डेटा के लिए एक समान फेस अलाइनमेंट (KYC और Live Scan दोनों में 100% समान)
    fun alignAndCropFaceBitmap(source: Bitmap, face: Face): Bitmap {
        var workingBmp = source
        var box = face.boundingBox

        val leftEye = face.getLandmark(FaceLandmark.LEFT_EYE)?.position
        val rightEye = face.getLandmark(FaceLandmark.RIGHT_EYE)?.position

        if (leftEye != null && rightEye != null) {
            val dy = (rightEye.y - leftEye.y).toDouble()
            val dx = (rightEye.x - leftEye.x).toDouble()
            val angleDeg = Math.toDegrees(atan2(dy, dx)).toFloat()
            if (abs(angleDeg) in 1.5f..28.0f) {
                val cx = box.exactCenterX()
                val cy = box.exactCenterY()
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
        val cx = boundingBox.centerX()
        val cy = boundingBox.centerY()
        val halfSide = (max(boundingBox.width(), boundingBox.height()) * 0.48f).toInt().coerceAtLeast(32)

        val left = (cx - halfSide).coerceIn(0, max(0, source.width - 1))
        val top = (cy - halfSide).coerceIn(0, max(0, source.height - 1))
        val right = (cx + halfSide).coerceIn(left + 1, source.width)
        val bottom = (cy + halfSide).coerceIn(top + 1, source.height)

        val cropped = Bitmap.createBitmap(source, left, top, right - left, bottom - top)
        return Bitmap.createScaledBitmap(cropped, 160, 160, true)
    }

    // ✅ KYC के समय 4-Pass Multi-Scale & Mirror Augmentation से उच्च-सटीकता वाला 192-D वेक्टर निकालें
    fun extractHighPrecisionKycEmbeddingList(alignedFaceBmp: Bitmap): List<Float> {
        val passes = mutableListOf<FloatArray>()

        // Pass 1: मूल अलाइन किया हुआ चेहरा
        computeEmbedding(alignedFaceBmp)?.let { passes.add(it) }

        // Pass 2: हॉरिजॉन्टल मिरर (फ्रंट और बैक कैमरे का अंतर खत्म करने के लिए सबसे ज़रूरी!)
        try {
            val flipMatrix = Matrix().apply { preScale(-1f, 1f) }
            val flipped = Bitmap.createBitmap(alignedFaceBmp, 0, 0, alignedFaceBmp.width, alignedFaceBmp.height, flipMatrix, true)
            computeEmbedding(flipped)?.let { passes.add(it) }
        } catch (e: Exception) {}

        // Pass 3: 4% टाइट सेंटर ज़ूम (कैमरा थोड़ा पास होने की स्थिति के लिए)
        try {
            val w = alignedFaceBmp.width
            val d = (w * 0.04f).toInt()
            val tight = Bitmap.createBitmap(alignedFaceBmp, d, d, w - d * 2, w - d * 2)
            computeEmbedding(tight)?.let { passes.add(it) }
        } catch (e: Exception) {}

        if (passes.isEmpty()) return emptyList()
        val merged = averageEmbeddings(passes)
        return merged.map { (it * 100000f).roundToInt() / 100000f }
    }

    fun computeEmbedding(faceBmp: Bitmap): FloatArray? {
        val interp = tfliteInterpreter ?: return null
        return try {
            val scaled = Bitmap.createScaledBitmap(faceBmp, inputImageSize, inputImageSize, true)
            val byteBuffer = ByteBuffer.allocateDirect(1 * inputImageSize * inputImageSize * 3 * 4)
            byteBuffer.order(ByteOrder.nativeOrder())

            val pixels = IntArray(inputImageSize * inputImageSize)
            scaled.getPixels(pixels, 0, inputImageSize, 0, 0, inputImageSize, inputImageSize)

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

    // Cosine Similarity (-1..1) को इंसानी समझ के लिए 0%..100% बायोमेट्रिक स्कोर में बदलें
    // Cosine 0.58 -> 70%, Cosine 0.70 -> 82%, Cosine 0.85 -> 95%
    private fun cosineToPercentage(cos: Float): Int {
        val mapped = ((cos + 0.12f) / 0.92f) * 100f
        return mapped.roundToInt().coerceIn(0, 99)
    }

    private fun buildStudentTemplate(student: Student): StudentBiometricTemplate? {
        val vectors = mutableListOf<FloatArray>()

        // 1. पहले सेव किए गए 192-D बायोमेट्रिक डेटा (stu.embeddings) का सीधा उपयोग करें
        if (student.embeddings.isNotEmpty()) {
            for (savedList in student.embeddings) {
                if (savedList.size == embeddingDim) {
                    val arr = FloatArray(embeddingDim)
                    for (i in 0 until embeddingDim) {
                        arr[i] = savedList[i]
                    }
                    vectors.add(l2Normalize(arr))
                }
            }
        }

        // 2. यदि किसी छात्र का 192-D वेक्टर अभी नहीं बना है, तो फोटो से बनाएं
        if (vectors.isEmpty() && student.photos.isNotEmpty()) {
            for (b64 in student.photos) {
                val bmp = decodeBase64Bitmap(b64)
                if (bmp != null) {
                    val emb = computeEmbedding(bmp)
                    if (emb != null) {
                        vectors.add(emb)
                    }
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
            return MatchResult(null, 999f, 0, 0f, false, emptyList())
        }

        // लाइव स्कैन में भी मूल + मिरर दोनों का औसत वेक्टर लें ताकि फ्रंट/बैक कैमरे का कोई फर्क न पड़े
        val livePasses = mutableListOf<FloatArray>()
        computeEmbedding(liveFaceBmp)?.let { livePasses.add(it) }
        try {
            val flipM = Matrix().apply { preScale(-1f, 1f) }
            val flippedLive = Bitmap.createBitmap(liveFaceBmp, 0, 0, liveFaceBmp.width, liveFaceBmp.height, flipM, true)
            computeEmbedding(flippedLive)?.let { livePasses.add(it) }
        } catch (e: Exception) {}

        if (livePasses.isEmpty()) {
            return MatchResult(null, 999f, 0, 0f, false, emptyList())
        }
        val liveVec = averageEmbeddings(livePasses)

        val scoredList = mutableListOf<Triple<Student, Float, Float>>() // Student, Distance, CosineSim

        for (stu in candidates) {
            if (stu.photos.isEmpty() && stu.embeddings.isEmpty()) continue
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
            val combinedCos = (maxPoseCos * 0.70f) + (centroidCos * 0.30f)
            val distance = (1f - combinedCos) * 10.0f

            scoredList.add(Triple(stu, distance, combinedCos))
        }

        if (scoredList.isEmpty()) {
            return MatchResult(null, 999f, 0, 0f, false, emptyList())
        }

        scoredList.sortByDescending { it.third } // उच्चतम Cosine Similarity सबसे ऊपर

        val bestStudent = scoredList[0].first
        val bestDistance = scoredList[0].second
        val bestCosine = scoredList[0].third
        val secondBestCosine = if (scoredList.size > 1) scoredList[1].third else 0f
        val cosineGap = bestCosine - secondBestCosine
        val bestPct = cosineToPercentage(bestCosine)

        val topCandidates = scoredList
            .filter { it.third >= AI_ASSIST_COSINE_MIN }
            .take(3)
            .map { it.first }

        val isHighConf = (bestCosine >= PASS_COSINE_THRESHOLD) &&
                (scoredList.size == 1 || cosineGap >= MIN_COSINE_MARGIN_GAP || bestCosine >= 0.68f)

        return if (isHighConf) {
            MatchResult(bestStudent, bestDistance, bestPct, bestCosine, true, topCandidates)
        } else if (topCandidates.isNotEmpty()) {
            MatchResult(bestStudent, bestDistance, bestPct, bestCosine, false, topCandidates)
        } else {
            MatchResult(null, bestDistance, bestPct, bestCosine, false, emptyList())
        }
    }

    fun findBestMatch(liveFaceBmp: Bitmap, candidates: List<Student>): Pair<Student?, Float> {
        val detailed = findBestMatchDetailed(liveFaceBmp, candidates)
        return Pair(detailed.student, detailed.distance)
    }

    // ✅ Online Gemini Vision AI Verification (उम्र, कपड़े, सिंदूर आदि की कोई शर्त नहीं—केवल 100% इंसान के चेहरे का मिलान)
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
            Bitmap.createScaledBitmap(liveFaceBmp, 220, 220, true)
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
                candidateDesc.append("Candidate $num: Name='${stu.name}'\n")
                partsArray.put(JSONObject().put("text", "CANDIDATE $num REFERENCE PHOTO:"))
                partsArray.put(JSONObject().put("inlineData", JSONObject().apply {
                    put("mimeType", "image/jpeg")
                    put("data", storedClean)
                }))
            }

            if (validList.isEmpty()) return null

            val strictPrompt = """
                You are an accurate biometric face verification engine.
                Compare the LIVE CAMERA FACE (Image 0) against the ${validList.size} reference photos:
                $candidateDesc
                
                RULES:
                1. Do NOT judge by age, clothing, or accessories—any person (child, adult, or elder) registered in the reference photo is valid.
                2. Compare ONLY facial bone structure, eyes, nose, mouth, and face geometry between Image 0 and each Candidate photo.
                3. If Image 0 is clearly the exact same person as one of the candidates, reply ONLY with that candidate number ("1", "2", or "3").
                4. If Image 0 is a different person from all candidates, reply ONLY with "NONE".
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
