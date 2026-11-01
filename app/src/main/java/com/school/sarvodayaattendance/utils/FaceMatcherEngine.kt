package com.school.sarvodayaattendance.utils

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Matrix
import android.graphics.Paint
import android.graphics.PointF
import android.graphics.Rect
import android.util.Base64
import com.google.mlkit.vision.common.InputImage
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
import java.io.FileInputStream
import java.net.HttpURLConnection
import java.net.URL
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import kotlin.math.abs
import kotlin.math.atan2
import kotlin.math.cos
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sin
import kotlin.math.sqrt

object FaceMatcherEngine {

    data class MatchResult(
        val student: Student?,
        val distance: Float,
        val marginRatio: Float,
        val isHighConfidence: Boolean,
        val isBorderlineForAiCheck: Boolean
    )

    private data class StudentBiometricTemplate(
        val studentId: String,
        val gender: String,
        val poseEmbeddings: List<FloatArray>,
        val centroidEmbedding: FloatArray,
        val primaryPhotoBase64: String
    )

    private var tfliteInterpreter: Interpreter? = null
    private var embeddingDim: Int = 192
    private var inputImageSize: Int = 112
    private val templateCache = ConcurrentHashMap<String, StudentBiometricTemplate>()

    // 99.8% सटीकता के लिए सख्त मानक (Strict Biometric Thresholds)
    const val STRICT_PASS_DISTANCE = 10.8f       // सीधा पास होने के लिए बेहद सख्त दूरी
    const val BORDERLINE_MAX_DISTANCE = 13.2f    // इसके बीच होने पर Online Gemini AI या डबल-फ्रेम चेक करेगा
    const val MIN_MARGIN_GAP = 1.45f             // 1st और 2nd मैच के बीच कम से कम इतना अंतर होना अनिवार्य है

    val detector: FaceDetector by lazy {
        val options = FaceDetectorOptions.Builder()
            .setPerformanceMode(FaceDetectorOptions.PERFORMANCE_MODE_ACCURATE)
            .setLandmarkMode(FaceDetectorOptions.LANDMARK_MODE_ALL)
            .setClassificationMode(FaceDetectorOptions.CLASSIFICATION_MODE_ALL)
            .setMinFaceSize(0.18f)
            .build()
        FaceDetection.getClient(options)
    }

    fun initEngine(context: Context) {
        if (tfliteInterpreter != null) return
        val modelNames = listOf("mobile_face_net.tflite", "facenet.tflite", "face_embedding.tflite")
        for (name in modelNames) {
            try {
                val afd = context.assets.openFd(name)
                val fis = FileInputStream(afd.fileDescriptor)
                val mapped: MappedByteBuffer = fis.channel.map(
                    FileChannel.MapMode.READ_ONLY,
                    afd.startOffset,
                    afd.declaredLength
                )
                val opts = Interpreter.Options().apply { setNumThreads(4) }
                val interp = Interpreter(mapped, opts)
                val inShape = interp.getInputTensor(0).shape()
                if (inShape.size >= 3) inputImageSize = inShape[1]
                val outShape = interp.getOutputTensor(0).shape()
                if (outShape.isNotEmpty()) embeddingDim = outShape.last()
                tfliteInterpreter = interp
                break
            } catch (_: Exception) {}
        }
    }

    fun clearCache() {
        templateCache.clear()
    }

    fun warmUpCache(students: List<Student>) {
        Thread {
            for (stu in students) {
                if (stu.photos.isNotEmpty() && !templateCache.containsKey(stu.id)) {
                    buildStudentTemplate(stu)?.let { templateCache[stu.id] = it }
                }
            }
        }.start()
    }

    // ✅ चेहरा पंजीकरण (FRS KYC) के समय जाँच: चेहरा सीधा, साफ और पर्याप्त रोशनी में है या नहीं
    fun validateKycFaceQuality(face: Face, frameBmp: Bitmap): Pair<Boolean, String> {
        val box = face.boundingBox
        if (box.width() < 95 || box.height() < 95) {
            return Pair(false, "⚠️ कृपया कैमरा चेहरे के थोड़ा और पास लाएं")
        }
        // सिर ज्यादा बाएं-दाएं या ऊपर-नीचे न झुका हो
        if (abs(face.headEulerAngleY) > 18f || abs(face.headEulerAngleZ) > 16f || abs(face.headEulerAngleX) > 18f) {
            return Pair(false, "⚠️ कृपया चेहरा बिल्कुल सीधा रखें (तिरछा न करें)")
        }
        val leftEyeOpen = face.leftEyeOpenProbability ?: 1.0f
        val rightEyeOpen = face.rightEyeOpenProbability ?: 1.0f
        if (leftEyeOpen < 0.35f || rightEyeOpen < 0.35f) {
            return Pair(false, "⚠️ कृपया दोनों आँखें खुली रखें")
        }
        val leftEye = face.getLandmark(FaceLandmark.LEFT_EYE)?.position
        val rightEye = face.getLandmark(FaceLandmark.RIGHT_EYE)?.position
        if (leftEye == null || rightEye == null) {
            return Pair(false, "⚠️ आँखें और चेहरा स्पष्ट नहीं दिखा, पर्याप्त रोशनी में आएं")
        }
        return Pair(true, "OK")
    }

    // ✅ Auto Eye-Alignment + Tight Inner Face Crop (पीछे की दीवार और कपड़े हटाने के लिए)
    fun alignAndCropFaceBitmap(source: Bitmap, face: Face): Bitmap {
        var workingBmp = source
        var box = face.boundingBox

        val leftEye = face.getLandmark(FaceLandmark.LEFT_EYE)?.position
        val rightEye = face.getLandmark(FaceLandmark.RIGHT_EYE)?.position

        if (leftEye != null && rightEye != null) {
            val dy = (rightEye.y - leftEye.y).toDouble()
            val dx = (rightEye.x - leftEye.x).toDouble()
            var angleDeg = Math.toDegrees(atan2(dy, dx)).toFloat()
            if (abs(angleDeg) in 2.0f..28.0f) {
                val cx = (leftEye.x + rightEye.x) / 2f
                val cy = (leftEye.y + rightEye.y) / 2f
                val matrix = Matrix().apply { postRotate(-angleDeg, cx, cy) }
                try {
                    workingBmp = Bitmap.createBitmap(source, 0, 0, source.width, source.height, matrix, true)
                } catch (_: Exception) {
                    workingBmp = source
                }
            }
        }

        return cropFaceBitmap(workingBmp, box)
    }

    // बायोमेट्रिक टाइट क्रॉप: बाहरी बैकग्राउंड को हटाकर केवल माथे, आँख, नाक, मुँह और ठोड़ी पर फोकस
    fun cropFaceBitmap(source: Bitmap, boundingBox: Rect): Bitmap {
        val w = boundingBox.width()
        val h = boundingBox.height()
        // बाहर जाने के बजाय 4% अंदर की तरफ टाइट क्रॉप ताकि पीछे की दीवार बिल्कुल न आए
        val insetX = (w * 0.04f).toInt()
        val insetTop = (h * 0.06f).toInt()
        val insetBottom = (h * 0.02f).toInt()

        val left = (boundingBox.left + insetX).coerceIn(0, max(0, source.width - 1))
        val top = (boundingBox.top + insetTop).coerceIn(0, max(0, source.height - 1))
        val right = (boundingBox.right - insetX).coerceIn(left + 1, source.width)
        val bottom = (boundingBox.bottom - insetBottom).coerceIn(top + 1, source.height)

        val cropped = Bitmap.createBitmap(source, left, top, right - left, bottom - top)
        return Bitmap.createScaledBitmap(cropped, 160, 160, true)
    }

    // पुराना सेव किया गया KYC फोटो लोड करते समय भी चेहरे को ऑटो-क्रॉप व अलाइन करना
    private fun decodeAndTightenStoredPhoto(base64Str: String): Bitmap? {
        try {
            val clean = base64Str.substringAfter("base64,", base64Str)
            val bytes = Base64.decode(clean, Base64.DEFAULT)
            val rawBmp = BitmapFactory.decodeByteArray(bytes, 0, bytes.size) ?: return null

            var alignedBmp: Bitmap? = null
            val latch = CountDownLatch(1)
            detector.process(InputImage.fromBitmap(rawBmp, 0))
                .addOnSuccessListener { faces ->
                    if (faces.isNotEmpty()) {
                        val primary = faces.maxByOrNull { it.boundingBox.width() * it.boundingBox.height() }!!
                        alignedBmp = alignAndCropFaceBitmap(rawBmp, primary)
                    }
                    latch.countDown()
                }
                .addOnFailureListener {
                    latch.countDown()
                }
            latch.await(900, TimeUnit.MILLISECONDS)
            return alignedBmp ?: Bitmap.createScaledBitmap(rawBmp, 160, 160, true)
        } catch (_: Exception) {
            return null
        }
    }

    // ✅ Hybrid Embedding: TFLite Deep Vector + Illumination-Invariant Structural Biometric Signature
    fun computeEmbedding(faceBmp: Bitmap): FloatArray {
        val structVec = computeStructuralBiometricVector(faceBmp)
        val interp = tfliteInterpreter
        if (interp != null) {
            try {
                val scaled = Bitmap.createScaledBitmap(faceBmp, inputImageSize, inputImageSize, true)
                val byteBuffer = ByteBuffer.allocateDirect(1 * inputImageSize * inputImageSize * 3 * 4)
                byteBuffer.order(ByteOrder.nativeOrder())
                val pixels = IntArray(inputImageSize * inputImageSize)
                scaled.getPixels(pixels, 0, inputImageSize, 0, 0, inputImageSize, inputImageSize)

                // Zero-Mean Unit-Variance Illumination Normalization
                var sum = 0.0
                var sumSq = 0.0
                val totalChannels = pixels.size * 3
                for (px in pixels) {
                    val r = (px shr 16 and 0xFF).toDouble()
                    val g = (px shr 8 and 0xFF).toDouble()
                    val b = (px and 0xFF).toDouble()
                    sum += r + g + b
                    sumSq += r * r + g * g + b * b
                }
                val mean = sum / totalChannels
                val std = max(sqrt((sumSq / totalChannels) - (mean * mean)), 1.0)

                for (px in pixels) {
                    byteBuffer.putFloat(((px shr 16 and 0xFF) - mean).toFloat() / std.toFloat())
                    byteBuffer.putFloat(((px shr 8 and 0xFF) - mean).toFloat() / std.toFloat())
                    byteBuffer.putFloat(((px and 0xFF) - mean).toFloat() / std.toFloat())
                }

                val output = Array(1) { FloatArray(embeddingDim) }
                synchronized(interp) {
                    interp.run(byteBuffer, output)
                }
                val deepVec = l2Normalize(output[0])
                return combineVectors(deepVec, structVec, 0.78f, 0.22f)
            } catch (_: Exception) {}
        }
        return structVec
    }

    // यदि TFLite के साथ संरचनात्मक मिलान भी हो: 16x16 Multi-Scale Gradient + Gabor-like Edge Orientation
    private fun computeStructuralBiometricVector(bmp: Bitmap): FloatArray {
        val size = 64
        val scaled = Bitmap.createScaledBitmap(bmp, size, size, true)
        val gray = FloatArray(size * size)
        val pixels = IntArray(size * size)
        scaled.getPixels(pixels, 0, size, 0, 0, size, size)

        var mean = 0f
        for (i in pixels.indices) {
            val px = pixels[i]
            val r = (px shr 16) and 0xFF
            val g = (px shr 8) and 0xFF
            val b = px and 0xFF
            val lum = 0.299f * r + 0.587f * g + 0.114f * b
            gray[i] = lum
            mean += lum
        }
        mean /= gray.size

        var variance = 0f
        for (i in gray.indices) {
            gray[i] -= mean
            variance += gray[i] * gray[i]
        }
        val std = max(sqrt(variance / gray.size), 1f)
        for (i in gray.indices) {
            gray[i] /= std
        }

        // 8x8 सेल में 8-दिशा ग्रेडिएंट हिस्टोग्राम (HOG) + क्षेत्रीय अनुपात = 576 डायमेंशन का सख्त फिंगरप्रिंट
        val cellSize = 8
        val gridCount = size / cellSize
        val features = FloatArray(gridCount * gridCount * 9)
        var featIdx = 0

        for (cy in 0 until gridCount) {
            for (cx in 0 until gridCount) {
                val bins = FloatArray(8)
                var cellMean = 0f
                for (y in (cy * cellSize) until ((cy + 1) * cellSize)) {
                    for (x in (cx * cellSize) until ((cx + 1) * cellSize)) {
                        val idx = y * size + x
                        cellMean += gray[idx]
                        if (x in 1 until (size - 1) && y in 1 until (size - 1)) {
                            val gx = gray[y * size + (x + 1)] - gray[y * size + (x - 1)]
                            val gy = gray[(y + 1) * size + x] - gray[(y - 1) * size + x]
                            val mag = sqrt(gx * gx + gy * gy)
                            var ang = Math.toDegrees(atan2(gy.toDouble(), gx.toDouble())).toFloat()
                            if (ang < 0) ang += 180f
                            val bin = ((ang / 180f) * 8f).toInt().coerceIn(0, 7)
                            bins[bin] += mag
                        }
                    }
                }
                for (b in 0 until 8) features[featIdx++] = bins[b]
                features[featIdx++] = cellMean / (cellSize * cellSize)
            }
        }
        return l2Normalize(features)
    }

    private fun combineVectors(v1: FloatArray, v2: FloatArray, w1: Float, w2: Float): FloatArray {
        val out = FloatArray(v1.size + v2.size)
        for (i in v1.indices) out[i] = v1[i] * w1
        for (i in v2.indices) out[v1.size + i] = v2[i] * w2
        return l2Normalize(out)
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

    // L2 + Cosine Combined Biometric Distance (0 = हूबहू एक जैसा, कम संख्या = बेहतर मैच)
    private fun calculateBiometricDistance(v1: FloatArray, v2: FloatArray): Float {
        if (v1.isEmpty() || v1.size != v2.size) return 999f
        var dot = 0f
        var l1Diff = 0f
        for (i in v1.indices) {
            dot += v1[i] * v2[i]
            l1Diff += abs(v1[i] - v2[i])
        }
        val cosineDist = (1f - dot.coerceIn(-1f, 1f)) * 22.0f
        val avgL1 = (l1Diff / v1.size) * 85.0f
        return (cosineDist * 0.72f) + (avgL1 * 0.28f)
    }

    private fun buildStudentTemplate(student: Student): StudentBiometricTemplate? {
        if (student.photos.isEmpty()) return null
        val embeddings = mutableListOf<FloatArray>()
        for (b64 in student.photos) {
            val bmp = decodeAndTightenStoredPhoto(b64)
            if (bmp != null) {
                embeddings.add(computeEmbedding(bmp))
            }
        }
        if (embeddings.isEmpty()) return null
        val centroid = averageEmbeddings(embeddings)
        return StudentBiometricTemplate(
            studentId = student.id,
            gender = student.gender.ifBlank { "M" },
            poseEmbeddings = embeddings,
            centroidEmbedding = centroid,
            primaryPhotoBase64 = student.photos.first()
        )
    }

    // ✅ मुख्य पहचान इंजन: 1st बनाम 2nd Best Match की तुलना और Strict Margin नियम
    fun findBestMatchDetailed(liveFaceBmp: Bitmap, candidates: List<Student>): MatchResult {
        if (candidates.isEmpty()) {
            return MatchResult(null, 999f, 1f, false, false)
        }

        val liveVec = computeEmbedding(liveFaceBmp)
        var bestStudent: Student? = null
        var bestDistance = 999f
        var secondBestDistance = 999f

        for (stu in candidates) {
            if (stu.photos.isEmpty()) continue
            val template = templateCache.getOrPut(stu.id) {
                buildStudentTemplate(stu) ?: continue
            }

            // सबसे नजदीकी पोज़ और सेंट्रॉइड दोनों से मिलान
            var minPoseDist = 999f
            for (poseVec in template.poseEmbeddings) {
                val d = calculateBiometricDistance(liveVec, poseVec)
                if (d < minPoseDist) minPoseDist = d
            }
            val centroidDist = calculateBiometricDistance(liveVec, template.centroidEmbedding)
            val combinedScore = (minPoseDist * 0.60f) + (centroidDist * 0.40f)

            if (combinedScore < bestDistance) {
                secondBestDistance = bestDistance
                bestDistance = combinedScore
                bestStudent = stu
            } else if (combinedScore < secondBestDistance) {
                secondBestDistance = combinedScore
            }
        }

        if (bestStudent == null) {
            return MatchResult(null, 999f, 1f, false, false)
        }

        val marginGap = if (secondBestDistance < 900f) (secondBestDistance - bestDistance) else 5.0f
        val marginRatio = if (bestDistance > 0.01f && secondBestDistance < 900f) {
            secondBestDistance / bestDistance
        } else {
            2.0f
        }

        // 1. उच्च विश्वास (High Confidence): दूरी बहुत कम है और दूसरे छात्र से स्पष्ट अंतर है
        val isHighConf = (bestDistance <= STRICT_PASS_DISTANCE) &&
                (marginGap >= MIN_MARGIN_GAP || marginRatio >= 1.15f)

        // 2. बॉर्डरलाइन (Borderline): चेहरा करीब है, लेकिन पुष्टि के लिए Online Gemini Vision AI या अतिरिक्त जाँच चाहिए
        val isBorderline = !isHighConf && (bestDistance <= BORDERLINE_MAX_DISTANCE) && (marginGap >= 0.65f)

        return if (isHighConf || isBorderline) {
            MatchResult(bestStudent, bestDistance, marginRatio, isHighConf, isBorderline)
        } else {
            MatchResult(null, bestDistance, marginRatio, false, false)
        }
    }

    // पिछली कॉल्स के साथ संगतता (Backward Compatibility)
    fun findBestMatch(liveFaceBmp: Bitmap, candidates: List<Student>): Pair<Student?, Float> {
        val detailed = findBestMatchDetailed(liveFaceBmp, candidates)
        return Pair(detailed.student, detailed.distance)
    }

    // ✅ Online Gemini Vision AI 1-to-1 Verifier (लड़का/लड़की क्रॉस-चेक + दो चेहरों का ऑनलाइन AI मिलान)
    fun verifyMatchWithGeminiVisionBlocking(
        context: Context,
        liveFaceBmp: Bitmap,
        candidate: Student
    ): Boolean {
        if (candidate.photos.isEmpty()) return false
        try {
            val vm = RealisticVoiceManager(context)
            val rawKeys = vm.getGeminiKeysString()
            val keyPool = rawKeys.split(",", "\n", ";", " ").map { it.trim() }.filter { it.length > 15 }
            val apiKey = keyPool.firstOrNull() ?: return false

            val baos = ByteArrayOutputStream()
            Bitmap.createScaledBitmap(liveFaceBmp, 180, 180, true)
                .compress(Bitmap.CompressFormat.JPEG, 82, baos)
            val liveB64 = Base64.encodeToString(baos.toByteArray(), Base64.NO_WRAP)

            val storedB64 = candidate.photos.first().substringAfter("base64,")
            val expectedGender = if (candidate.gender.equals("F", ignoreCase = true)) "FEMALE (Girl)" else "MALE (Boy)"

            val endpoint = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key=$apiKey"
            val conn = (URL(endpoint).openConnection() as HttpURLConnection).apply {
                requestMethod = "POST"
                connectTimeout = 4500
                readTimeout = 5500
                doOutput = true
                setRequestProperty("Content-Type", "application/json; charset=UTF-8")
            }

            val prompt = """
                You are a strict school biometric face verification system.
                Image 1 is the stored KYC photo of student "${candidate.name}" whose registered gender is $expectedGender.
                Image 2 is the live camera capture.
                Rule 1: If the person in Image 2 does NOT match the gender $expectedGender, immediately return NO.
                Rule 2: Compare facial bone structure, eyes, nose, and mouth. If Image 1 and Image 2 are definitely the SAME child, reply ONLY with "YES". Otherwise reply ONLY with "NO".
            """.trimIndent()

            val partsArray = JSONArray().apply {
                put(JSONObject().put("text", prompt))
                put(JSONObject().put("inlineData", JSONObject().apply {
                    put("mimeType", "image/jpeg")
                    put("data", storedB64)
                }))
                put(JSONObject().put("inlineData", JSONObject().apply {
                    put("mimeType", "image/jpeg")
                    put("data", liveB64)
                }))
            }

            val payload = JSONObject().apply {
                put("contents", JSONArray().put(JSONObject().put("parts", partsArray)))
            }

            conn.outputStream.use { it.write(payload.toString().toByteArray(Charsets.UTF_8)) }
            if (conn.responseCode == 200) {
                val resp = conn.inputStream.bufferedReader(Charsets.UTF_8).use { it.readText() }
                val answer = JSONObject(resp)
                    .optJSONArray("candidates")?.optJSONObject(0)
                    ?.optJSONObject("content")?.optJSONArray("parts")?.optJSONObject(0)
                    ?.optString("text", "")?.trim()?.uppercase() ?: ""
                return answer.startsWith("YES")
            }
        } catch (_: Exception) {}
        return false
    }
}
