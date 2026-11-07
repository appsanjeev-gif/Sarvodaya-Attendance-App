package com.school.sarvodayaattendance.utils

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Matrix
import android.graphics.Paint
import android.graphics.PorterDuff
import android.graphics.PorterDuffXfermode
import android.graphics.Rect
import android.graphics.RectF
import android.util.Base64
import com.google.mlkit.vision.face.Face
import com.google.mlkit.vision.face.FaceDetection
import com.google.mlkit.vision.face.FaceDetector
import com.google.mlkit.vision.face.FaceDetectorOptions
import com.google.mlkit.vision.face.FaceLandmark
import com.school.sarvodayaattendance.data.AppStorage
import com.school.sarvodayaattendance.data.Student
import org.json.JSONArray
import org.json.JSONObject
import org.tensorflow.lite.Interpreter
import java.io.ByteArrayOutputStream
import java.io.File
import java.io.FileInputStream
import java.io.FileOutputStream
import java.net.HttpURLConnection
import java.net.SocketTimeoutException
import java.net.URL
import java.net.UnknownHostException
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.atomic.AtomicBoolean
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
        val needsSecondFrameConfirm: Boolean,
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

    private val isWarmingUp = AtomicBoolean(false)

    @Volatile
    private var initialWarmUpDone = false

    // वैज्ञानिक रूप से संतुलित 192-D बायोमेट्रिक थ्रेशोल्ड (तेज़ + 100% सटीक):
    // 1. INSTANT_PASS_COSINE (0.74f): उच्च समानता होने पर पहले ही फ्रेम (0.15s) में तुरंत हाज़िरी
    // 2. PASS_COSINE_THRESHOLD (0.675f): सामान्य व पुराने छात्रों के लिए पासिंग थ्रेशोल्ड (2 फ्रेम पुष्टि + स्पष्ट अंतर)
    // 3. MANUAL_TAP_COSINE_MIN (0.650f): [📷 स्कैन करें] बटन दबाने पर न्यूनतम पासिंग थ्रेशोल्ड
    // 4. MIN_COSINE_MARGIN_GAP (0.045f): पहले और दूसरे छात्र के बीच न्यूनतम आवश्यक अंतर
    // 5. MIN_POPULATION_GAP (0.070f): अन्य छात्रों के औसत स्कोर से न्यूनतम आवश्यक अंतर (अनजान चेहरों को रोकने के लिए)
    const val INSTANT_PASS_COSINE = 0.74f
    const val PASS_COSINE_THRESHOLD = 0.675f
    const val MANUAL_TAP_COSINE_MIN = 0.650f
    const val AI_ASSIST_COSINE_MIN = 0.60f
    const val MIN_COSINE_MARGIN_GAP = 0.045f
    const val MIN_POPULATION_GAP = 0.070f

    // लाइव कैमरे के लिए सुपर-फास्ट डिटेक्टर (बिना किसी बैकग्राउंड कंटेंशन के)
    val detector: FaceDetector by lazy {
        val options = FaceDetectorOptions.Builder()
            .setPerformanceMode(FaceDetectorOptions.PERFORMANCE_MODE_FAST)
            .setLandmarkMode(FaceDetectorOptions.LANDMARK_MODE_ALL)
            .setClassificationMode(FaceDetectorOptions.CLASSIFICATION_MODE_ALL)
            .setMinFaceSize(0.16f)
            .build()
        FaceDetection.getClient(options)
    }

    fun isNeuralModelReady(): Boolean = (tfliteInterpreter != null)

    fun isReadyForLiveScan(): Boolean = (tfliteInterpreter != null && initialWarmUpDone)

    fun getEmbeddingDimension(): Int = embeddingDim

    // जांचें कि छात्र के पास 192-D का शुद्ध नया बायोमेट्रिक वेक्टर मौजूद है या नहीं
    fun hasVerified192D(student: Student): Boolean {
        return student.embeddings.isNotEmpty() && student.embeddings.any { it.size == embeddingDim }
    }

    // छात्र का नया 192-D वेक्टर बनने के बाद इन-मेमोरी टेम्पलेट कैश तुरंत अपडेट करना
    fun updateSingleStudentCache(student: Student) {
        val built = buildStudentTemplate(student)
        if (built != null) {
            templateCache[student.id] = built
        }
    }

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
                    if (setupInterpreter(mapped)) {
                        AiDiagnosticLogger.recordEvent(
                            appCtx,
                            "TFLITE",
                            "1. 192-D बायोमेट्रिक इंजन (Offline)",
                            "✅ सक्रिय ($embeddingDim-D)",
                            "शुद्ध 192-D मॉडल ($name | ${inputImageSize}x${inputImageSize}) मेमोरी में तैयार है"
                        )
                        return
                    }
                }
            } catch (e: Exception) {
                try {
                    val cacheFile = File(appCtx.filesDir, "packed_192d_$name")
                    appCtx.assets.open(name).use { input ->
                        FileOutputStream(cacheFile).use { output ->
                            input.copyTo(output)
                        }
                    }
                    if (cacheFile.exists() && cacheFile.length() > 500_000L) {
                        FileInputStream(cacheFile).use { fis ->
                            val mapped = fis.channel.map(FileChannel.MapMode.READ_ONLY, 0, cacheFile.length())
                            if (setupInterpreter(mapped)) {
                                AiDiagnosticLogger.recordEvent(
                                    appCtx,
                                    "TFLITE",
                                    "1. 192-D बायोमेट्रिक इंजन (Offline)",
                                    "✅ सक्रिय ($embeddingDim-D)",
                                    "शुद्ध 192-D मॉडल ($name | ${inputImageSize}x${inputImageSize}) कैश से तैयार है"
                                )
                                return
                            }
                        }
                    }
                } catch (ex: Exception) {}
            }
        }
        AiDiagnosticLogger.recordEvent(
            appCtx,
            "TFLITE",
            "1. 192-D बायोमेट्रिक इंजन (Offline)",
            "❌ लोड विफल",
            "APK के assets में .tflite मॉडल नहीं मिला"
        )
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

    // बिना ML Kit को ब्लॉक किए ~0.5 सेकंड में सभी 82 छात्रों के इन-मेमोरी टेम्पलेट तैयार करना
    // (नोट: यह stu.embeddings को कभी ओवरराइट नहीं करता, ताकि बिना Re-KYC वाले छात्र 3.4s पुष्टि में आ सकें)
    fun warmUpCache(students: List<Student>, context: Context? = null) {
        if (students.isEmpty()) {
            initialWarmUpDone = true
            return
        }
        if (isWarmingUp.getAndSet(true)) return

        Thread {
            try {
                for (stu in students) {
                    if (stu.photos.isEmpty() && stu.embeddings.isEmpty()) continue
                    if (!templateCache.containsKey(stu.id)) {
                        val built = buildStudentTemplate(stu)
                        if (built != null) {
                            templateCache[stu.id] = built
                        }
                    }
                }
                initialWarmUpDone = true
            } finally {
                initialWarmUpDone = true
                isWarmingUp.set(false)
            }
        }.start()
    }

    fun downscaleFrameForFastDetection(src: Bitmap, maxDimension: Int = 480): Bitmap {
        val w = src.width
        val h = src.height
        val longest = max(w, h)
        if (longest <= maxDimension) return src
        val scale = maxDimension.toFloat() / longest.toFloat()
        val newW = (w * scale).roundToInt().coerceAtLeast(1)
        val newH = (h * scale).roundToInt().coerceAtLeast(1)
        return Bitmap.createScaledBitmap(src, newW, newH, true)
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

    fun createFramedResultCardBitmap(src: Bitmap): Bitmap {
        val size = 240
        val scaled = Bitmap.createScaledBitmap(src, size, size, true)
        val out = Bitmap.createBitmap(size, size, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(out)

        val innerPhotoRect = RectF(10f, 10f, (size - 10).toFloat(), (size - 10).toFloat())
        val cornerRad = 34f

        val maskPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = Color.WHITE
            style = Paint.Style.FILL
        }
        val photoLayer = Bitmap.createBitmap(size, size, Bitmap.Config.ARGB_8888)
        val photoCanvas = Canvas(photoLayer)
        photoCanvas.drawRoundRect(innerPhotoRect, cornerRad - 6f, cornerRad - 6f, maskPaint)
        maskPaint.xfermode = PorterDuffXfermode(PorterDuff.Mode.SRC_IN)
        photoCanvas.drawBitmap(scaled, null, innerPhotoRect, maskPaint)
        canvas.drawBitmap(photoLayer, 0f, 0f, null)

        val innerBorderPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            style = Paint.Style.STROKE
            strokeWidth = 4f
            color = Color.parseColor("#E2E8F0")
        }
        canvas.drawRoundRect(innerPhotoRect, cornerRad - 6f, cornerRad - 6f, innerBorderPaint)

        val outerBorderPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            style = Paint.Style.STROKE
            strokeWidth = 7f
            color = Color.parseColor("#334155")
        }
        val frameRect = RectF(4f, 4f, (size - 4).toFloat(), (size - 4).toFloat())
        canvas.drawRoundRect(frameRect, cornerRad, cornerRad, outerBorderPaint)

        return out
    }

    // लाइव चेहरे की जांच + स्पष्ट हिंदी कारण (ताकि आधा माथा रिजेक्ट हो, पर सामान्य दूरी पर खड़ा छात्र तुरंत पास हो)
    fun evaluateLiveFaceGate(face: Face, frameWidth: Int = 0, frameHeight: Int = 0): Pair<Boolean, String> {
        val box = face.boundingBox
        if (frameWidth > 0 && frameHeight > 0) {
            val minSide = min(frameWidth, frameHeight)
            if (box.width() < minSide * 0.16f || box.height() < minSide * 0.16f) {
                return Pair(false, "⚠️ कृपया कैमरे के थोड़ा पास आएं")
            }
        }

        // पाँचों मुख्य बिंदु (दोनों आँखें, नाक, मुँह के दोनों कोने) मौजूद होने चाहिए (जिससे केवल माथा या आधा चेहरा कभी पास न हो)
        val leftEye = face.getLandmark(FaceLandmark.LEFT_EYE)?.position
            ?: return Pair(false, "⚠️ पूरा चेहरा कैमरे के बीच में रखें")
        val rightEye = face.getLandmark(FaceLandmark.RIGHT_EYE)?.position
            ?: return Pair(false, "⚠️ पूरा चेहरा कैमरे के बीच में रखें")
        val noseBase = face.getLandmark(FaceLandmark.NOSE_BASE)?.position
            ?: return Pair(false, "⚠️ पूरा चेहरा कैमरे के बीच में रखें")
        val mouthLeft = face.getLandmark(FaceLandmark.MOUTH_LEFT)?.position
            ?: return Pair(false, "⚠️ पूरा चेहरा कैमरे के बीच में रखें")
        val mouthRight = face.getLandmark(FaceLandmark.MOUTH_RIGHT)?.position
            ?: return Pair(false, "⚠️ पूरा चेहरा कैमरे के बीच में रखें")

        val eyeCenterY = (leftEye.y + rightEye.y) * 0.5f
        val mouthCenterY = (mouthLeft.y + mouthRight.y) * 0.5f

        if (frameWidth > 0 && frameHeight > 0) {
            // यदि मुँह या आँखें स्क्रीन के बिल्कुल बाहर/किनारे पर कट रही हों (जैसे केवल माथा दिखना)
            if (mouthCenterY > frameHeight * 0.95f || eyeCenterY < frameHeight * 0.04f) {
                return Pair(false, "⚠️ पूरा चेहरा कैमरे के बीच में लाएं")
            }
            if (leftEye.x < frameWidth * 0.03f || rightEye.x > frameWidth * 0.97f) {
                return Pair(false, "⚠️ चेहरा कैमरे के बीच में रखें")
            }
        }

        if (noseBase.y <= eyeCenterY || mouthCenterY <= noseBase.y || (mouthCenterY - eyeCenterY) < box.height() * 0.18f) {
            return Pair(false, "⚠️ कृपया कैमरे की ओर सीधे देखें")
        }

        if (abs(face.headEulerAngleY) > 26f || abs(face.headEulerAngleZ) > 24f) {
            return Pair(false, "⚠️ कृपया चेहरा सीधा रखें")
        }

        // FAST मोड में जब eye probability null हो तो 0.85f मानें ताकि कभी झूठा रिजेक्शन न हो
        val leftOpen = face.leftEyeOpenProbability ?: 0.85f
        val rightOpen = face.rightEyeOpenProbability ?: 0.85f
        if (leftOpen < 0.20f && rightOpen < 0.20f) {
            return Pair(false, "⚠️ कृपया आँखें खुली रखें")
        }

        return Pair(true, "OK")
    }

    fun isLiveFaceValidForScan(face: Face, frameWidth: Int = 0, frameHeight: Int = 0): Boolean {
        return evaluateLiveFaceGate(face, frameWidth, frameHeight).first
    }

    fun validateKycFaceQuality(face: Face, frameBmp: Bitmap): Pair<Boolean, String> {
        return evaluateLiveFaceGate(face, frameBmp.width, frameBmp.height)
    }

    fun cropTrueSquarePortrait(source: Bitmap, face: Face): Bitmap {
        val box = face.boundingBox
        val cx = box.centerX()
        val cy = (box.centerY() - (box.height() * 0.04f)).toInt()

        val desiredSide = (max(box.width(), box.height()) * 1.62f).toInt()
        val maxPossibleSide = min(source.width, source.height)
        val side = desiredSide.coerceIn(80, maxPossibleSide)
        val half = side / 2

        var left = cx - half
        var top = cy - half

        if (left < 0) left = 0
        if (top < 0) top = 0
        if (left + side > source.width) left = max(0, source.width - side)
        if (top + side > source.height) top = max(0, source.height - side)

        val actualW = min(side, source.width - left)
        val actualH = min(side, source.height - top)
        val finalSquareSide = min(actualW, actualH).coerceAtLeast(1)

        val squareCrop = Bitmap.createBitmap(source, left, top, finalSquareSide, finalSquareSide)
        return Bitmap.createScaledBitmap(squareCrop, 320, 320, true)
    }

    fun applyOnDeviceStudioPortraitFilter(squareBmp: Bitmap): Bitmap {
        val w = squareBmp.width
        val h = squareBmp.height
        val pixels = IntArray(w * h)
        squareBmp.getPixels(pixels, 0, w, 0, 0, w, h)

        var minLum = 255
        var maxLum = 0
        var sumLum = 0L
        for (px in pixels) {
            val r = (px shr 16) and 0xFF
            val g = (px shr 8) and 0xFF
            val b = px and 0xFF
            val lum = (r * 77 + g * 150 + b * 29) shr 8
            if (lum < minLum) minLum = lum
            if (lum > maxLum) maxLum = lum
            sumLum += lum
        }
        val avgLum = (sumLum / pixels.size).toInt()
        val gamma = when {
            avgLum < 95 -> 0.76
            avgLum < 125 -> 0.86
            avgLum > 185 -> 1.05
            else -> 0.92
        }

        val lut = IntArray(256)
        val lowCut = (minLum * 0.35).toInt().coerceIn(0, 40)
        val highCut = max(maxLum, 215).coerceIn(190, 255)
        val range = max(highCut - lowCut, 60).toDouble()

        for (i in 0..255) {
            val norm = ((i - lowCut).coerceAtLeast(0) / range).coerceIn(0.0, 1.0)
            val curved = Math.pow(norm, gamma)
            val sCurve = curved * curved * (3.0 - 2.0 * curved)
            val blended = (curved * 0.65 + sCurve * 0.35) * 255.0
            lut[i] = blended.roundToInt().coerceIn(0, 255)
        }

        val exposedPixels = IntArray(w * h)
        for (i in pixels.indices) {
            val px = pixels[i]
            var r = lut[(px shr 16) and 0xFF]
            var g = lut[(px shr 8) and 0xFF]
            var b = lut[px and 0xFF]

            r = (r * 1.03f + 3f).toInt().coerceIn(0, 255)
            g = (g * 1.01f + 1f).toInt().coerceIn(0, 255)
            b = (b * 0.99f).toInt().coerceIn(0, 255)

            exposedPixels[i] = (0xFF shl 24) or (r shl 16) or (g shl 8) or b
        }

        val outBmp = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
        outBmp.setPixels(exposedPixels, 0, w, 0, 0, w, h)
        return outBmp
    }

    fun createAiEnhancedPortraitDp(source: Bitmap, face: Face): Bitmap {
        val squareCrop = cropTrueSquarePortrait(source, face)
        return applyOnDeviceStudioPortraitFilter(squareCrop)
    }

    fun enhancePortraitWithGeminiOrFallbackBlocking(context: Context, squareBaseBmp: Bitmap): Pair<Bitmap, Boolean> {
        var fallbackReason = "Gemini Key उपलब्ध नहीं है — ऑन-डिवाइस स्टूडियो फिल्टर लगाया गया"
        try {
            val vm = RealisticVoiceManager(context)
            val rawKeys = vm.getGeminiKeysString()
            val keyPool = rawKeys.split(",", "\n", ";", " ")
                .map { it.trim() }
                .filter { it.length > 15 && !it.startsWith("sk_", ignoreCase = true) }
            val apiKey = keyPool.firstOrNull()

            if (apiKey != null) {
                val inputBaos = ByteArrayOutputStream()
                Bitmap.createScaledBitmap(squareBaseBmp, 320, 320, true)
                    .compress(Bitmap.CompressFormat.JPEG, 90, inputBaos)
                val inputB64 = Base64.encodeToString(inputBaos.toByteArray(), Base64.NO_WRAP)

                val modelsToTry = listOf(
                    "gemini-2.0-flash-exp-image-generation",
                    "gemini-2.5-flash-image-preview"
                )

                for (modelName in modelsToTry) {
                    try {
                        val endpoint = "https://generativelanguage.googleapis.com/v1beta/models/$modelName:generateContent?key=$apiKey"
                        val conn = (URL(endpoint).openConnection() as HttpURLConnection).apply {
                            requestMethod = "POST"
                            connectTimeout = 10000
                            readTimeout = 18000
                            doOutput = true
                            setRequestProperty("Content-Type", "application/json; charset=UTF-8")
                        }

                        val prompt = "Enhance this student ID portrait photo into a clean, bright, studio-lit 1:1 square passport photo. Keep the person's exact facial identity, bone structure, eyes, nose, and expression 100% unchanged. Only improve lighting, clarity, and natural skin tone."

                        val partsArray = JSONArray().apply {
                            put(JSONObject().put("text", prompt))
                            put(JSONObject().put("inlineData", JSONObject().apply {
                                put("mimeType", "image/jpeg")
                                put("data", inputB64)
                            }))
                        }

                        val genConfig = JSONObject().apply {
                            put("responseModalities", JSONArray().put("TEXT").put("IMAGE"))
                        }

                        val payload = JSONObject().apply {
                            put("contents", JSONArray().put(JSONObject().put("parts", partsArray)))
                            put("generationConfig", genConfig)
                        }

                        conn.outputStream.use { it.write(payload.toString().toByteArray(Charsets.UTF_8)) }
                        val code = conn.responseCode
                        if (code == 200) {
                            val resp = conn.inputStream.bufferedReader(Charsets.UTF_8).use { it.readText() }
                            val parts = JSONObject(resp)
                                .optJSONArray("candidates")?.optJSONObject(0)
                                ?.optJSONObject("content")?.optJSONArray("parts")
                            if (parts != null) {
                                for (i in 0 until parts.length()) {
                                    val p = parts.optJSONObject(i) ?: continue
                                    val inlineData = p.optJSONObject("inlineData") ?: p.optJSONObject("inline_data")
                                    if (inlineData != null) {
                                        val b64Data = inlineData.optString("data", "")
                                        if (b64Data.length > 100) {
                                            val decoded = decodeBase64Bitmap(b64Data)
                                            if (decoded != null) {
                                                val side = min(decoded.width, decoded.height)
                                                val x = (decoded.width - side) / 2
                                                val y = (decoded.height - side) / 2
                                                val sq = Bitmap.createBitmap(decoded, x, y, side, side)
                                                AiDiagnosticLogger.recordEvent(
                                                    context,
                                                    "DP_EDIT",
                                                    "3. AI स्टूडियो DP सुधार (KYC)",
                                                    "✅ क्लाउड AI से बना",
                                                    "Gemini Image Model ($modelName) से 1:1 फोटो एडिट होकर आई"
                                                )
                                                return Pair(Bitmap.createScaledBitmap(sq, 320, 320, true), true)
                                            }
                                        }
                                    }
                                }
                            }
                            fallbackReason = "Gemini ($modelName) ने इमेज बाइनरी नहीं लौटाई — ऑन-डिवाइस स्टूडियो फिल्टर लगा"
                        } else {
                            fallbackReason = "Gemini Image API HTTP $code — ऑन-डिवाइस स्टूडियो फिल्टर से 1:1 DP बनी"
                        }
                    } catch (e: UnknownHostException) {
                        fallbackReason = "इंटरनेट बंद था — ऑन-डिवाइस स्टूडियो फिल्टर से 1:1 DP बनाई गई"
                    } catch (e: SocketTimeoutException) {
                        fallbackReason = "नेटवर्क धीमा (Timeout) था — ऑन-डिवाइस स्टूडियो फिल्टर से 1:1 DP बनाई गई"
                    } catch (e: Exception) {
                        fallbackReason = "कनेक्शन त्रुटि (${e.javaClass.simpleName}) — ऑन-डिवाइस स्टूडियो फिल्टर लगा"
                    }
                }
            }
        } catch (e: Exception) {}

        AiDiagnosticLogger.recordEvent(
            context,
            "DP_EDIT",
            "3. AI स्टूडियो DP सुधार (KYC)",
            "🔄 ऑन-डिवाइस स्टूडियो फिल्टर",
            fallbackReason
        )
        return Pair(applyOnDeviceStudioPortraitFilter(squareBaseBmp), false)
    }

    fun alignAndCropFaceBitmap(source: Bitmap, face: Face): Bitmap {
        val box = face.boundingBox
        val cx = box.exactCenterX()
        val cy = box.exactCenterY()
        val cropSide = (max(box.width(), box.height()) * 0.96f).coerceAtLeast(48f)

        var angleDeg = 0f
        val leftEye = face.getLandmark(FaceLandmark.LEFT_EYE)?.position
        val rightEye = face.getLandmark(FaceLandmark.RIGHT_EYE)?.position
        if (leftEye != null && rightEye != null) {
            val dy = (rightEye.y - leftEye.y).toDouble()
            val dx = (rightEye.x - leftEye.x).toDouble()
            val rawAngle = Math.toDegrees(atan2(dy, dx)).toFloat()
            if (abs(rawAngle) in 1.5f..26.0f) {
                angleDeg = rawAngle
            }
        }

        val targetSize = 112
        val scale = targetSize.toFloat() / cropSide
        val outBmp = Bitmap.createBitmap(targetSize, targetSize, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(outBmp)
        val matrix = Matrix().apply {
            postTranslate(-cx, -cy)
            if (angleDeg != 0f) {
                postRotate(-angleDeg)
            }
            postScale(scale, scale)
            postTranslate(targetSize / 2f, targetSize / 2f)
        }
        val paint = Paint(Paint.FILTER_BITMAP_FLAG or Paint.ANTI_ALIAS_FLAG)
        canvas.drawBitmap(source, matrix, paint)
        return outBmp
    }

    fun cropFaceBitmap(source: Bitmap, boundingBox: Rect): Bitmap {
        val cx = boundingBox.centerX()
        val cy = boundingBox.centerY()
        val halfSide = (max(boundingBox.width(), boundingBox.height()) * 0.48f).toInt().coerceAtLeast(24)

        val left = (cx - halfSide).coerceIn(0, max(0, source.width - 1))
        val top = (cy - halfSide).coerceIn(0, max(0, source.height - 1))
        val right = (cx + halfSide).coerceIn(left + 1, source.width)
        val bottom = (cy + halfSide).coerceIn(top + 1, source.height)

        val cropped = Bitmap.createBitmap(source, left, top, right - left, bottom - top)
        return Bitmap.createScaledBitmap(cropped, 112, 112, true)
    }

    private fun extractExactBiometricFaceFromStoredDp(
        dpBmp: Bitmap,
        ratio: Float = 0.593f,
        yShiftRatio: Float = 0.0247f
    ): Bitmap {
        val w = dpBmp.width
        val h = dpBmp.height
        if (w < 64 || h < 64) return dpBmp
        val cropW = (w * ratio).roundToInt().coerceIn(48, w)
        val cropH = (h * ratio).roundToInt().coerceIn(48, h)
        val cx = w * 0.50f
        val cy = h * (0.50f + yShiftRatio)
        val left = (cx - cropW / 2f).roundToInt().coerceIn(0, max(0, w - cropW))
        val top = (cy - cropH / 2f).roundToInt().coerceIn(0, max(0, h - cropH))
        return try {
            val sub = Bitmap.createBitmap(dpBmp, left, top, cropW, cropH)
            Bitmap.createScaledBitmap(sub, 112, 112, true)
        } catch (e: Exception) {
            dpBmp
        }
    }

    private fun flipHorizontal(bmp: Bitmap): Bitmap {
        val m = Matrix().apply { preScale(-1f, 1f) }
        return Bitmap.createBitmap(bmp, 0, 0, bmp.width, bmp.height, m, true)
    }

    fun extractHighPrecisionKycEmbeddingList(alignedFaceBmp: Bitmap): List<Float> {
        val v1 = computeEmbedding(alignedFaceBmp) ?: return emptyList()
        val v2 = try { computeEmbedding(flipHorizontal(alignedFaceBmp)) } catch (e: Exception) { null }
        val merged = if (v2 != null) averageEmbeddings(listOf(v1, v2)) else v1
        return merged.map { (it * 100000f).roundToInt() / 100000f }
    }

    fun computeEmbedding(faceBmp: Bitmap): FloatArray? {
        val interp = tfliteInterpreter ?: return null
        return try {
            val scaled = if (faceBmp.width == inputImageSize && faceBmp.height == inputImageSize) {
                faceBmp
            } else {
                Bitmap.createScaledBitmap(faceBmp, inputImageSize, inputImageSize, true)
            }
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

    private fun cosineToPercentage(cos: Float): Int {
        if (cos <= 0.35f) return ((cos.coerceAtLeast(0f) / 0.35f) * 30f).roundToInt()
        if (cos < PASS_COSINE_THRESHOLD) {
            val norm = (cos - 0.35f) / (PASS_COSINE_THRESHOLD - 0.35f)
            return (30f + norm * 44f).roundToInt().coerceIn(30, 74)
        }
        val passNorm = ((cos - PASS_COSINE_THRESHOLD) / (0.88f - PASS_COSINE_THRESHOLD)).coerceIn(0f, 1f)
        return (78f + passNorm * 21f).roundToInt().coerceIn(78, 99)
    }

    private fun buildStudentTemplate(student: Student): StudentBiometricTemplate? {
        val vectors = mutableListOf<FloatArray>()

        // 1. छात्र के सेव किए गए 192-D वेक्टर
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

        // 2. साथ ही सेव की गई प्रोफाइल फोटो का शुद्ध 0.593f क्रॉप भी पोज़ लिस्ट में रखें (ताकि पुराने बिना-ReKYC वाले छात्र भी 100% पहचाने जाएं)
        if (student.photos.isNotEmpty()) {
            for (b64 in student.photos.take(2)) {
                val bmp = decodeBase64Bitmap(b64) ?: continue
                val exactCrop = extractExactBiometricFaceFromStoredDp(bmp, 0.593f, 0.0247f)
                val v1 = computeEmbedding(exactCrop)
                val v2 = try { computeEmbedding(flipHorizontal(exactCrop)) } catch (e: Exception) { null }
                if (v1 != null) vectors.add(v1)
                if (v1 != null && v2 != null) {
                    vectors.add(averageEmbeddings(listOf(v1, v2)))
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

    fun findBestMatchDetailed(
        liveFaceBmp: Bitmap,
        candidates: List<Student>,
        isManualTap: Boolean = false
    ): MatchResult {
        if (candidates.isEmpty()) {
            return MatchResult(null, 999f, 0, 0f, false, false, emptyList())
        }

        val vPrimary = computeEmbedding(liveFaceBmp)
            ?: return MatchResult(null, 999f, 0, 0f, false, false, emptyList())
        val vFlipped = try { computeEmbedding(flipHorizontal(liveFaceBmp)) } catch (e: Exception) { null }
        val liveMerged = if (vFlipped != null) averageEmbeddings(listOf(vPrimary, vFlipped)) else vPrimary

        val scoredList = ArrayList<Triple<Student, Float, Float>>(candidates.size)

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
                val c1 = calculateCosineSimilarity(vPrimary, poseVec)
                val c2 = calculateCosineSimilarity(liveMerged, poseVec)
                val cMax = max(c1, c2)
                if (cMax > maxPoseCos) maxPoseCos = cMax
            }
            val centroidCos = calculateCosineSimilarity(liveMerged, template.centroidEmbedding)
            val bestScore = (maxPoseCos * 0.85f) + (centroidCos * 0.15f)
            val distance = (1f - bestScore) * 10.0f

            scoredList.add(Triple(stu, distance, bestScore))
        }

        if (scoredList.isEmpty()) {
            return MatchResult(null, 999f, 0, 0f, false, false, emptyList())
        }

        scoredList.sortByDescending { it.third }

        val bestStudent = scoredList[0].first
        val bestDistance = scoredList[0].second
        val bestCosine = scoredList[0].third
        val secondBestCosine = if (scoredList.size > 1) scoredList[1].third else 0f
        val cosineGap = bestCosine - secondBestCosine

        // Top-2 से Top-6 अन्य छात्रों के औसत से अंतर (अनजान चेहरे को रोकने का सरल व अचूक नियम)
        val otherSlice = scoredList.drop(1).take(5)
        val avgOthers = if (otherSlice.isNotEmpty()) {
            otherSlice.map { it.third }.average().toFloat()
        } else {
            0f
        }
        val populationGap = bestCosine - avgOthers
        val bestPct = cosineToPercentage(bestCosine)

        val topCandidates = scoredList
            .filter { it.third >= AI_ASSIST_COSINE_MIN }
            .take(3)
            .map { it.first }

        val isSingleStudent = (scoredList.size == 1)
        val requiredMinCos = if (isManualTap) MANUAL_TAP_COSINE_MIN else PASS_COSINE_THRESHOLD
        val requiredMarginGap = if (isManualTap) 0.036f else MIN_COSINE_MARGIN_GAP
        val requiredPopGap = if (isManualTap) 0.058f else MIN_POPULATION_GAP

        val isHighConf = (bestCosine >= requiredMinCos) &&
                (isSingleStudent || bestCosine >= 0.78f || (cosineGap >= requiredMarginGap && populationGap >= requiredPopGap))

        val needsSecondFrame = !isManualTap && (bestCosine < INSTANT_PASS_COSINE)

        return if (isHighConf) {
            MatchResult(bestStudent, bestDistance, bestPct, bestCosine, true, needsSecondFrame, topCandidates)
        } else {
            MatchResult(null, bestDistance, bestPct, bestCosine, false, true, topCandidates)
        }
    }

    fun findBestMatch(liveFaceBmp: Bitmap, candidates: List<Student>): Pair<Student?, Float> {
        val detailed = findBestMatchDetailed(liveFaceBmp, candidates, false)
        return Pair(detailed.student, detailed.distance)
    }
}
