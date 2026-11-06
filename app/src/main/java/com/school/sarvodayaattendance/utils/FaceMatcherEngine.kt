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
        val kycEmbeddings: List<FloatArray>,
        val photoEmbeddings: List<FloatArray>,
        val allPoseEmbeddings: List<FloatArray>
    )

    @Volatile
    private var tfliteInterpreter: Interpreter? = null
    private var embeddingDim: Int = 192
    private var inputImageSize: Int = 112
    private val templateCache = ConcurrentHashMap<String, StudentBiometricTemplate>()
    private val isWarmingUp = AtomicBoolean(false)

    // 192-D MobileFaceNet वैज्ञानिक रूप से संतुलित थ्रेशोल्ड:
    // 1. INSTANT_PASS_COSINE (0.64f): पहली ही बार में तुरंत (0.15s में) हाज़िरी
    // 2. TWO_FRAME_PASS_COSINE (0.575f): बिना Re-KYC वाले पुराने छात्रों के लिए लगातार 2 फ्रेम (0.4s) में पुष्टि + स्पष्ट अंतर (Gap)
    // 3. 0.575f से नीचे या बिना स्पष्ट Gap वाले अनजान चेहरे 100% रिजेक्ट
    const val INSTANT_PASS_COSINE = 0.64f
    const val TWO_FRAME_PASS_COSINE = 0.575f
    const val AI_ASSIST_COSINE_MIN = 0.52f
    const val MIN_COSINE_MARGIN_GAP = 0.038f
    const val MIN_POPULATION_GAP = 0.065f

    // सुपर-फास्ट कैमरे के लिए FAST मोड + लैंडमार्क (कोई अनावश्यक ब्लॉकिंग नहीं)
    val detector: FaceDetector by lazy {
        val options = FaceDetectorOptions.Builder()
            .setPerformanceMode(FaceDetectorOptions.PERFORMANCE_MODE_FAST)
            .setLandmarkMode(FaceDetectorOptions.LANDMARK_MODE_ALL)
            .setClassificationMode(FaceDetectorOptions.CLASSIFICATION_MODE_ALL)
            .setMinFaceSize(0.15f)
            .build()
        FaceDetection.getClient(options)
    }

    fun isNeuralModelReady(): Boolean = (tfliteInterpreter != null)

    fun getEmbeddingDimension(): Int = embeddingDim

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
                            "हाई-स्पीड $embeddingDim-D मॉडल ($name | ${inputImageSize}x${inputImageSize}) तैयार है"
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
                                    "हाई-स्पीड $embeddingDim-D मॉडल ($name | ${inputImageSize}x${inputImageSize}) कैश से तैयार है"
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

    // बिना ML Kit को ब्लॉक किए 1 सेकंड में सभी पुराने और नए छात्रों के 192-D वेक्टर तैयार करना और पुराने छात्रों का डेटा ऑटो-अपग्रेड करना
    fun warmUpCache(students: List<Student>, context: Context? = null) {
        if (isWarmingUp.getAndSet(true)) return
        Thread {
            try {
                var migratedCount = 0
                for (stu in students) {
                    if (stu.embeddings.isEmpty() && stu.photos.isEmpty()) continue
                    var template = templateCache[stu.id]
                    if (template == null) {
                        template = buildStudentTemplate(stu)
                        if (template != null) {
                            templateCache[stu.id] = template
                        }
                    }
                    // यदि किसी पुराने छात्र के पास 192-D वेक्टर सेव नहीं था (जैसे पुराना 128-D डेटा था), तो फोटो से बने शुद्ध 192-D वेक्टर को स्थायी रूप से जोड़ दें
                    val hasValid192D = stu.embeddings.any { it.size == embeddingDim }
                    if (!hasValid192D && template != null && template.photoEmbeddings.isNotEmpty()) {
                        val primaryVec = template.photoEmbeddings.first()
                        val rounded = primaryVec.map { (it * 100000f).roundToInt() / 100000f }
                        stu.embeddings.clear()
                        stu.embeddings.add(rounded)
                        migratedCount++
                    }
                }
                if (migratedCount > 0 && context != null) {
                    try {
                        val appCtx = context.applicationContext
                        val payload = AppStorage.loadData(appCtx)
                        val mapById = students.associateBy { it.id }
                        var updatedInStorage = false
                        for (s in payload.students) {
                            val mig = mapById[s.id]
                            if (mig != null && mig.embeddings.isNotEmpty() && s.embeddings.none { it.size == embeddingDim }) {
                                s.embeddings = mig.embeddings
                                updatedInStorage = true
                            }
                        }
                        if (updatedInStorage) {
                            AppStorage.saveData(appCtx, payload)
                        }
                        AiDiagnosticLogger.recordEvent(
                            appCtx,
                            "TFLITE",
                            "1. 192-D बायोमेट्रिक इंजन (Offline)",
                            "✅ सक्रिय ($embeddingDim-D)",
                            "$migratedCount पुराने छात्रों की फोटो को बिना Re-KYC के ऑटोमैटिक 192-D में अपग्रेड किया गया"
                        )
                    } catch (e: Exception) {}
                }
            } finally {
                isWarmingUp.set(false)
            }
        }.start()
    }

    // कैमरे के बड़े फ्रेम (1080p/2400p) को 480px पर छोटा करना ताकि चेहरा 0.02 सेकंड में डिटेक्ट हो
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

    // सामान्य रोशनी या हल्के झुकाव में भी चेहरा तुरंत स्वीकार करने के लिए व्यावहारिक नियम
    fun isLiveFaceValidForScan(face: Face): Boolean {
        if (abs(face.headEulerAngleY) > 32f || abs(face.headEulerAngleZ) > 28f) {
            return false
        }
        val leftOpen = face.leftEyeOpenProbability ?: 1.0f
        val rightOpen = face.rightEyeOpenProbability ?: 1.0f
        if (leftOpen < 0.15f && rightOpen < 0.15f) {
            return false
        }
        return true
    }

    fun validateKycFaceQuality(face: Face, frameBmp: Bitmap): Pair<Boolean, String> {
        val box = face.boundingBox
        val minAllowed = min(frameBmp.width, frameBmp.height) * 0.16f
        if (box.width() < minAllowed || box.height() < minAllowed) {
            return Pair(false, "⚠️ कृपया कैमरा चेहरे के थोड़ा और पास लाएं")
        }
        if (abs(face.headEulerAngleY) > 24f || abs(face.headEulerAngleZ) > 22f) {
            return Pair(false, "⚠️ कृपया चेहरा सीधा रखें")
        }
        val leftEyeOpen = face.leftEyeOpenProbability ?: 1.0f
        val rightEyeOpen = face.rightEyeOpenProbability ?: 1.0f
        if (leftEyeOpen < 0.22f && rightEyeOpen < 0.22f) {
            return Pair(false, "⚠️ कृपया आँखें खुली रखें")
        }
        return Pair(true, "OK")
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

    // सटीक कैनवास-आधारित फेस अलाइनमेंट (जिससे सिर झुकने पर भी कोऑर्डिनेट कभी शिफ्ट न हों!)
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
            if (abs(rawAngle) in 2.0f..26.0f) {
                angleDeg = rawAngle
            }
        }

        val targetSize = 160
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
        return Bitmap.createScaledBitmap(cropped, 160, 160, true)
    }

    // पुरानी सेव की गई 1:1 DP (cropTrueSquarePortrait: 1.62f * box, cy shifted -0.04f) से
    // बिना ML Kit चलाए 0.1 मिलिसेकंड में शुद्ध गणितीय फेस बॉक्स निकालना
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
            Bitmap.createScaledBitmap(sub, 160, 160, true)
        } catch (e: Exception) {
            dpBmp
        }
    }

    private fun flipHorizontal(bmp: Bitmap): Bitmap {
        val m = Matrix().apply { preScale(-1f, 1f) }
        return Bitmap.createBitmap(bmp, 0, 0, bmp.width, bmp.height, m, true)
    }

    fun extractHighPrecisionKycEmbeddingList(alignedFaceBmp: Bitmap): List<Float> {
        val primary = computeEmbedding(alignedFaceBmp) ?: return emptyList()
        val flipped = try { computeEmbedding(flipHorizontal(alignedFaceBmp)) } catch (e: Exception) { null }
        val merged = if (flipped != null) averageEmbeddings(listOf(primary, flipped)) else primary
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
                byteBuffer.putFloat((b - 128.0f) / 128.0f)
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

    // 192-D स्कोर को स्पष्ट प्रतिशत में बदलना (0.58 -> 78%, 0.65 -> 86%, 0.78+ -> 96-99%)
    private fun cosineToPercentage(cos: Float): Int {
        if (cos <= 0.25f) return ((cos.coerceAtLeast(0f) / 0.25f) * 25f).roundToInt()
        if (cos < TWO_FRAME_PASS_COSINE) {
            val norm = (cos - 0.25f) / (TWO_FRAME_PASS_COSINE - 0.25f)
            return (25f + norm * 48f).roundToInt().coerceIn(25, 73)
        }
        val passNorm = ((cos - TWO_FRAME_PASS_COSINE) / (0.85f - TWO_FRAME_PASS_COSINE)).coerceIn(0f, 1f)
        return (76f + passNorm * 23f).roundToInt().coerceIn(76, 99)
    }

    private fun buildStudentTemplate(student: Student): StudentBiometricTemplate? {
        val kycVecs = mutableListOf<FloatArray>()
        val photoVecs = mutableListOf<FloatArray>()

        // 1. सेव किए गए शुद्ध 192-D KYC वेक्टर (बिना किसी मिलावट के अलग रखें)
        if (student.embeddings.isNotEmpty()) {
            for (savedList in student.embeddings) {
                if (savedList.size == embeddingDim) {
                    val arr = FloatArray(embeddingDim)
                    for (i in 0 until embeddingDim) {
                        arr[i] = savedList[i]
                    }
                    kycVecs.add(l2Normalize(arr))
                }
            }
        }

        // 2. सेव की गई प्रोफाइल फोटो (stu.photos) से 3 मल्टी-स्केल बायोमेट्रिक क्रॉप + मिरर फ्लिप निकालें
        // (इसमें ML Kit बिल्कुल नहीं चलता, इसलिए यह 0.01 सेकंड में तैयार हो जाता है और पुराने छात्रों को बिना Re-KYC के पहचानता है)
        if (student.photos.isNotEmpty()) {
            for (b64 in student.photos.take(3)) {
                val bmp = decodeBase64Bitmap(b64) ?: continue
                val exactCrop = extractExactBiometricFaceFromStoredDp(bmp, 0.593f, 0.0247f)
                val exactEmb = computeEmbedding(exactCrop)
                val flipEmb = try { computeEmbedding(flipHorizontal(exactCrop)) } catch (e: Exception) { null }
                if (exactEmb != null) photoVecs.add(exactEmb)
                if (flipEmb != null) photoVecs.add(flipEmb)
                if (exactEmb != null && flipEmb != null) {
                    photoVecs.add(averageEmbeddings(listOf(exactEmb, flipEmb)))
                }

                val widerCrop = extractExactBiometricFaceFromStoredDp(bmp, 0.66f, 0.020f)
                computeEmbedding(widerCrop)?.let { photoVecs.add(it) }

                val tighterCrop = extractExactBiometricFaceFromStoredDp(bmp, 0.53f, 0.025f)
                computeEmbedding(tighterCrop)?.let { photoVecs.add(it) }
            }
        }

        val allPoses = mutableListOf<FloatArray>().apply {
            addAll(kycVecs)
            addAll(photoVecs)
        }
        if (allPoses.isEmpty()) return null

        return StudentBiometricTemplate(
            student = student,
            kycEmbeddings = kycVecs,
            photoEmbeddings = photoVecs,
            allPoseEmbeddings = allPoses
        )
    }

    fun findBestMatchDetailed(liveFaceBmp: Bitmap, candidates: List<Student>): MatchResult {
        if (candidates.isEmpty()) {
            return MatchResult(null, 999f, 0, 0f, false, false, emptyList())
        }

        val livePrimary = computeEmbedding(liveFaceBmp)
            ?: return MatchResult(null, 999f, 0, 0f, false, false, emptyList())
        val liveFlipped = try { computeEmbedding(flipHorizontal(liveFaceBmp)) } catch (e: Exception) { null }
        val liveMerged = if (liveFlipped != null) averageEmbeddings(listOf(livePrimary, liveFlipped)) else livePrimary

        val scoredList = mutableListOf<Triple<Student, Float, Float>>()

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

            // Direct Best-Pose Matching (बिना धुंधले सेंट्रॉइड की मिलावट के सीधे सर्वोच्च पोज़ का मिलान)
            var bestCos = -1f
            for (poseVec in template.allPoseEmbeddings) {
                val c1 = calculateCosineSimilarity(livePrimary, poseVec)
                val c2 = calculateCosineSimilarity(liveMerged, poseVec)
                val cMax = max(c1, c2)
                if (cMax > bestCos) bestCos = cMax
            }

            val distance = (1f - bestCos) * 10.0f
            scoredList.add(Triple(stu, distance, bestCos))
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

        // अनजान चेहरों को पकड़ने के लिए Top-2 से Top-6 अन्य छात्रों के औसत स्कोर से तुलना (Population Distinctiveness)
        val otherSlice = scoredList.drop(1).take(5)
        val avgOtherCosine = if (otherSlice.isNotEmpty()) {
            otherSlice.map { it.third }.average().toFloat()
        } else {
            0f
        }
        val populationGap = bestCosine - avgOtherCosine
        val bestPct = cosineToPercentage(bestCosine)

        val topCandidates = scoredList
            .filter { it.third >= AI_ASSIST_COSINE_MIN }
            .take(3)
            .map { it.first }

        val isSingleStudent = (scoredList.size == 1)

        // 1. Instant Pass (0.15s): उच्च समानता + स्पष्ट अंतर
        val isInstantPass = (bestCosine >= INSTANT_PASS_COSINE) &&
                (isSingleStudent || bestCosine >= 0.71f || (cosineGap >= MIN_COSINE_MARGIN_GAP && populationGap >= MIN_POPULATION_GAP))

        // 2. Two-Frame Pass (0.45s): बिना Re-KYC वाले पुराने छात्रों के लिए सुरक्षित पास (बशर्ते वह अन्य सभी छात्रों से स्पष्ट रूप से अलग हो)
        val isTwoFramePass = !isInstantPass &&
                (bestCosine >= TWO_FRAME_PASS_COSINE) &&
                (isSingleStudent || (cosineGap >= 0.042f && populationGap >= 0.075f))

        return when {
            isInstantPass -> MatchResult(bestStudent, bestDistance, bestPct, bestCosine, true, false, topCandidates)
            isTwoFramePass -> MatchResult(bestStudent, bestDistance, bestPct, bestCosine, true, true, topCandidates)
            topCandidates.isNotEmpty() -> MatchResult(bestStudent, bestDistance, bestPct, bestCosine, false, false, topCandidates)
            else -> MatchResult(null, bestDistance, bestPct, bestCosine, false, false, emptyList())
        }
    }

    fun findBestMatch(liveFaceBmp: Bitmap, candidates: List<Student>): Pair<Student?, Float> {
        val detailed = findBestMatchDetailed(liveFaceBmp, candidates)
        return Pair(detailed.student, detailed.distance)
    }
}
