package com.school.sarvodayaattendance.utils

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.ColorMatrix
import android.graphics.ColorMatrixColorFilter
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
                    if (setupInterpreter(mapped)) {
                        AiDiagnosticLogger.recordEvent(
                            appCtx,
                            "TFLITE",
                            "1. 192-D बायोमेट्रिक इंजन (Offline)",
                            "✅ सक्रिय ($embeddingDim-D)",
                            "मॉडल ($name) मेमोरी में सफलतापूर्वक लोड है"
                        )
                        return
                    }
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
                            if (setupInterpreter(mapped)) {
                                AiDiagnosticLogger.recordEvent(
                                    appCtx,
                                    "TFLITE",
                                    "1. 192-D बायोमेट्रिक इंजन (Offline)",
                                    "✅ सक्रिय ($embeddingDim-D)",
                                    "मॉडल ($name) कैश मेमोरी से लोड है"
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

    fun cropTrueSquarePortrait(source: Bitmap, face: Face): Bitmap {
        val box = face.boundingBox
        val cx = box.centerX()
        val cy = (box.centerY() - (box.height() * 0.04f)).toInt()

        val desiredSide = (max(box.width(), box.height()) * 1.62f).toInt()
        val maxPossibleSide = min(source.width, source.height)
        val side = desiredSide.coerceIn(100, maxPossibleSide)
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

        val finalPixels = exposedPixels.clone()
        val thresholdDiff = 26

        for (y in 1 until h - 1) {
            for (x in 1 until w - 1) {
                val idx = y * w + x
                val cPx = exposedPixels[idx]
                val cr = (cPx shr 16) and 0xFF
                val cg = (cPx shr 8) and 0xFF
                val cb = cPx and 0xFF

                var sumR = cr * 4
                var sumG = cg * 4
                var sumB = cb * 4
                var weightSum = 4

                val neighbors = intArrayOf(
                    exposedPixels[idx - 1],
                    exposedPixels[idx + 1],
                    exposedPixels[idx - w],
                    exposedPixels[idx + w]
                )

                var edgeDiff = 0
                for (nPx in neighbors) {
                    val nr = (nPx shr 16) and 0xFF
                    val ng = (nPx shr 8) and 0xFF
                    val nb = nPx and 0xFF
                    val diff = (abs(cr - nr) + abs(cg - ng) + abs(cb - nb)) / 3
                    edgeDiff += diff
                    if (diff < thresholdDiff) {
                        sumR += nr
                        sumG += ng
                        sumB += nb
                        weightSum++
                    }
                }

                val outR: Int
                val outG: Int
                val outB: Int
                if (edgeDiff > 55) {
                    val avgN_R = ((neighbors[0] shr 16 and 0xFF) + (neighbors[1] shr 16 and 0xFF) + (neighbors[2] shr 16 and 0xFF) + (neighbors[3] shr 16 and 0xFF)) shr 2
                    val avgN_G = ((neighbors[0] shr 8 and 0xFF) + (neighbors[1] shr 8 and 0xFF) + (neighbors[2] shr 8 and 0xFF) + (neighbors[3] shr 8 and 0xFF)) shr 2
                    val avgN_B = ((neighbors[0] and 0xFF) + (neighbors[1] and 0xFF) + (neighbors[2] and 0xFF) + (neighbors[3] and 0xFF)) shr 2
                    outR = (cr + ((cr - avgN_R) * 0.28f).toInt()).coerceIn(0, 255)
                    outG = (cg + ((cg - avgN_G) * 0.28f).toInt()).coerceIn(0, 255)
                    outB = (cb + ((cb - avgN_B) * 0.28f).toInt()).coerceIn(0, 255)
                } else {
                    outR = (sumR / weightSum).coerceIn(0, 255)
                    outG = (sumG / weightSum).coerceIn(0, 255)
                    outB = (sumB / weightSum).coerceIn(0, 255)
                }

                finalPixels[idx] = (0xFF shl 24) or (outR shl 16) or (outG shl 8) or outB
            }
        }

        val outBmp = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
        outBmp.setPixels(finalPixels, 0, w, 0, 0, w, h)
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
                            connectTimeout = 15000
                            readTimeout = 30000
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
                            fallbackReason = "Gemini Image API HTTP $code (फ्री टियर पर इमेज जनरेशन सीमित) — ऑन-डिवाइस स्टूडियो फिल्टर लगा"
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
        var workingBmp = source
        val box = face.boundingBox

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

    fun extractHighPrecisionKycEmbeddingList(alignedFaceBmp: Bitmap): List<Float> {
        val passes = mutableListOf<FloatArray>()

        computeEmbedding(alignedFaceBmp)?.let { passes.add(it) }

        try {
            val flipMatrix = Matrix().apply { preScale(-1f, 1f) }
            val flipped = Bitmap.createBitmap(alignedFaceBmp, 0, 0, alignedFaceBmp.width, alignedFaceBmp.height, flipMatrix, true)
            computeEmbedding(flipped)?.let { passes.add(it) }
        } catch (e: Exception) {}

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

    private fun cosineToPercentage(cos: Float): Int {
        val mapped = ((cos + 0.12f) / 0.92f) * 100f
        return mapped.roundToInt().coerceIn(0, 99)
    }

    private fun buildStudentTemplate(student: Student): StudentBiometricTemplate? {
        val vectors = mutableListOf<FloatArray>()

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

        scoredList.sortByDescending { it.third }

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
            val apiKey = keyPool.firstOrNull()
            if (apiKey == null) {
                AiDiagnosticLogger.recordEvent(
                    context,
                    "VISION_VERIFY",
                    "2. AI फेस मिलान सहायता (Gemini Vision)",
                    "⚠️ Key उपलब्ध नहीं",
                    "कन्फ्यूजन की स्थिति में Gemini Key न होने से ऑनलाइन वेरिफिकेशन नहीं हो सका"
                )
                return null
            }

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
            val code = conn.responseCode
            if (code == 200) {
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
                    val matched = validList[chosenIdx]
                    AiDiagnosticLogger.recordEvent(
                        context,
                        "VISION_VERIFY",
                        "2. AI फेस मिलान सहायता (Gemini Vision)",
                        "✅ पहचान सफल",
                        "Gemini AI ने संशय दूर कर '${matched.name}' की पुष्टि की"
                    )
                    return matched
                } else {
                    AiDiagnosticLogger.recordEvent(
                        context,
                        "VISION_VERIFY",
                        "2. AI फेस मिलान सहायता (Gemini Vision)",
                        "ℹ️ अज्ञात चेहरा (NONE)",
                        "Gemini AI ने जाँच कर बताया कि यह चेहरा किसी उम्मीदवार से मैच नहीं करता"
                    )
                }
            } else {
                AiDiagnosticLogger.recordEvent(
                    context,
                    "VISION_VERIFY",
                    "2. AI फेस मिलान सहायता (Gemini Vision)",
                    "❌ सर्वर त्रुटि ($code)",
                    "Gemini Vision API रिस्पॉन्स कोड: $code"
                )
            }
        } catch (e: UnknownHostException) {
            AiDiagnosticLogger.recordEvent(
                context,
                "VISION_VERIFY",
                "2. AI फेस मिलान सहायता (Gemini Vision)",
                "❌ इंटरनेट बंद है",
                "मोबाइल डेटा बंद होने से ऑनलाइन क्रॉस-चेक नहीं हो सका"
            )
        } catch (e: SocketTimeoutException) {
            AiDiagnosticLogger.recordEvent(
                context,
                "VISION_VERIFY",
                "2. AI फेस मिलान सहायता (Gemini Vision)",
                "⚠️ नेटवर्क धीमा (Timeout)",
                "इंटरनेट स्लो होने से 6 सेकंड में जवाब नहीं आया"
            )
        } catch (e: Exception) {
            AiDiagnosticLogger.recordEvent(
                context,
                "VISION_VERIFY",
                "2. AI फेस मिलान सहायता (Gemini Vision)",
                "❌ त्रुटि",
                "कारण: ${e.javaClass.simpleName}"
            )
        }
        return null
    }
}
