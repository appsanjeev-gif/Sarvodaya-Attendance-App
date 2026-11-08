package com.school.sarvodayaattendance.utils

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Matrix
import android.graphics.Paint
import android.graphics.PointF
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
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.min
import kotlin.math.pow
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

    // सख्त 192-D KYC सर्कल एनीमेशन के लिए लाइव प्रोग्रेस डेटा (बिना किसी टेक्स्ट या नोटिफिकेशन के)
    data class StrictKycProgress(
        val progressFraction: Float, // 0.0f से 1.0f
        val isPerfectFrame: Boolean  // जब 5° कोण, दोनों कान, सीधी नाक और खुली आँखें 100% सटीक हों
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

    // शुद्ध 192-D बायोमेट्रिक थ्रेशोल्ड (सुपर-फास्ट और सटीक पहचान):
    const val INSTANT_PASS_COSINE = 0.73f
    const val PASS_COSINE_THRESHOLD = 0.67f
    const val MANUAL_TAP_COSINE_MIN = 0.64f
    const val AI_ASSIST_COSINE_MIN = 0.58f
    const val MIN_COSINE_MARGIN_GAP = 0.042f
    const val MIN_POPULATION_GAP = 0.065f

    // सख्त KYC मानक (5° कोण सीमा + दोनों कान बराबर + नाक सीधी + आँखें कैमरे पर):
    const val STRICT_KYC_MAX_ANGLE_DEG = 5.0f
    const val STRICT_KYC_MIN_EAR_SYMMETRY = 0.76f
    const val STRICT_KYC_MIN_EYE_OPEN = 0.68f

    // लाइव कैमरे के लिए सुपर-फास्ट डिटेक्टर (सभी लैंडमार्क व क्लासिफिकेशन सहित)
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
                            "शुद्ध $embeddingDim-D मॉडल ($name | ${inputImageSize}x${inputImageSize}) मेमोरी में तैयार है"
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
                                    "शुद्ध $embeddingDim-D मॉडल ($name | ${inputImageSize}x${inputImageSize}) कैश से तैयार है"
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

    // केवल शुद्ध 192-D वेक्टर को मेमोरी में लोड करना (0.01 सेकंड में सुपर-फास्ट स्टार्टअप)
    fun warmUpCache(students: List<Student>, context: Context? = null) {
        if (students.isEmpty()) {
            initialWarmUpDone = true
            return
        }
        if (isWarmingUp.getAndSet(true)) return

        Thread {
            try {
                val appCtx = context?.applicationContext
                if (appCtx != null) {
                    val legacyPrefs = listOf(
                        "SarvodayaBiometricAlignCacheV6",
                        "SarvodayaCleanBiometricV7",
                        "SarvodayaVerified192DKycV8",
                        "SarvodayaOldPhotoVecCacheV8"
                    )
                    for (prefName in legacyPrefs) {
                        try {
                            appCtx.getSharedPreferences(prefName, Context.MODE_PRIVATE).edit().clear().apply()
                        } catch (e: Exception) {}
                    }
                }

                var removedLegacy128D = false
                var ready192DCount = 0

                for (stu in students) {
                    val beforeSize = stu.embeddings.size
                    stu.embeddings.removeAll { it.size != embeddingDim }
                    if (stu.embeddings.size != beforeSize) {
                        removedLegacy128D = true
                    }

                    if (stu.embeddings.isEmpty()) {
                        templateCache.remove(stu.id)
                        continue
                    }

                    val built = buildStudentTemplate(stu)
                    if (built != null) {
                        templateCache[stu.id] = built
                        ready192DCount++
                    }
                }

                initialWarmUpDone = true

                if (appCtx != null) {
                    if (removedLegacy128D) {
                        try {
                            val masterPayload = AppStorage.loadData(appCtx)
                            for (s in masterPayload.students) {
                                s.embeddings.removeAll { it.size != embeddingDim }
                            }
                            AppStorage.saveData(appCtx, masterPayload)
                        } catch (e: Exception) {}
                    }
                    AiDiagnosticLogger.recordEvent(
                        appCtx,
                        "TFLITE",
                        "1. 192-D बायोमेट्रिक इंजन (Offline)",
                        "✅ सक्रिय ($embeddingDim-D)",
                        "शुद्ध 192-D इंजन तैयार (192-D KYC पूर्ण: $ready192DCount/${students.size} छात्र)"
                    )
                }
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

    // दैनिक उपस्थिति (Live Attendance Scan) के लिए व्यावहारिक फेस गेट
    fun evaluateLiveFaceGate(face: Face, frameWidth: Int = 0, frameHeight: Int = 0): Pair<Boolean, String> {
        val box = face.boundingBox
        if (frameWidth > 0 && frameHeight > 0) {
            val minSide = min(frameWidth, frameHeight)
            if (box.width() < minSide * 0.16f || box.height() < minSide * 0.16f) {
                return Pair(false, "⚠️ कृपया कैमरे के थोड़ा पास आएं")
            }
        }

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

        if (abs(face.headEulerAngleY) > 24f || abs(face.headEulerAngleZ) > 22f) {
            return Pair(false, "⚠️ कृपया चेहरा सीधा रखें")
        }

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

    // ★ सख्त 192-D KYC सर्कल मूल्यांकन (बिना किसी टेक्स्ट के केवल गोल लाइन भरने के लिए):
    // शर्तें:
    // 1. दोनों कान (LEFT_EAR व RIGHT_EAR) दिखने अनिवार्य और दोनों की नाक से दूरी बराबर अनुपात में हो।
    // 2. सिर का घुमाव (Yaw Y, Roll Z, Pitch X) सख्त 5.0° के भीतर हो।
    // 3. नाक (NOSE_BASE) दोनों आँखों के ठीक बीचों-बीच सीधी हो और मुँह के केंद्र के ठीक ऊपर हो।
    // 4. दोनों आँखें कैमरे पर खुली हों (>= 0.68f) और एक सीधी क्षैतिज रेखा में हों।
    // 5. चेहरा सर्कल के केंद्र में हो और नीचे कंधों के लिए पर्याप्त जगह हो।
    fun evaluateStrictKycProgress(face: Face?, frameWidth: Int, frameHeight: Int): StrictKycProgress {
        if (face == null || frameWidth <= 0 || frameHeight <= 0) {
            return StrictKycProgress(0f, false)
        }

        val box = face.boundingBox
        val minSide = min(frameWidth, frameHeight).toFloat()
        val faceSizeRatio = min(box.width(), box.height()) / minSide

        // 1. चेहरे का आकार और फ्रेम के बीच में स्थिति (कंधे आने के लिए नीचे स्पेस सहित)
        val sizeOk = faceSizeRatio in 0.22f..0.68f
        val cxRatio = box.exactCenterX() / frameWidth.toFloat()
        val cyRatio = box.exactCenterY() / frameHeight.toFloat()
        val centerOk = (cxRatio in 0.32f..0.68f) && (cyRatio in 0.24f..0.64f)

        val sizeAndCenterScore = when {
            sizeOk && centerOk -> 0.20f
            faceSizeRatio >= 0.16f && (cxRatio in 0.22f..0.78f) && (cyRatio in 0.16f..0.75f) -> 0.11f
            else -> 0.04f
        }

        // 2. मुख्य लैंडमार्क (आँखें, नाक, मुँह)
        val leftEye: PointF = face.getLandmark(FaceLandmark.LEFT_EYE)?.position
            ?: return StrictKycProgress(sizeAndCenterScore, false)
        val rightEye: PointF = face.getLandmark(FaceLandmark.RIGHT_EYE)?.position
            ?: return StrictKycProgress(sizeAndCenterScore, false)
        val noseBase: PointF = face.getLandmark(FaceLandmark.NOSE_BASE)?.position
            ?: return StrictKycProgress(sizeAndCenterScore, false)
        val mouthLeft: PointF = face.getLandmark(FaceLandmark.MOUTH_LEFT)?.position
            ?: return StrictKycProgress(sizeAndCenterScore, false)
        val mouthRight: PointF = face.getLandmark(FaceLandmark.MOUTH_RIGHT)?.position
            ?: return StrictKycProgress(sizeAndCenterScore, false)

        // 3. आँखें कैमरे पर खुली और सीधी रेखा में
        val leftOpen = face.leftEyeOpenProbability ?: 0f
        val rightOpen = face.rightEyeOpenProbability ?: 0f
        val eyeLineAngleDeg = abs(
            Math.toDegrees(
                atan2(
                    (rightEye.y - leftEye.y).toDouble(),
                    abs(rightEye.x - leftEye.x).toDouble().coerceAtLeast(1.0)
                )
            ).toFloat()
        )
        val eyesPerfect = (leftOpen >= STRICT_KYC_MIN_EYE_OPEN) &&
                (rightOpen >= STRICT_KYC_MIN_EYE_OPEN) &&
                (eyeLineAngleDeg <= 4.5f)
        val eyeScore = when {
            eyesPerfect -> 0.20f
            leftOpen >= 0.45f && rightOpen >= 0.45f && eyeLineAngleDeg <= 8.0f -> 0.11f
            else -> 0.03f
        }

        // 4. नाक एकदम सीधी और दोनों आँखों व मुँह के ठीक केंद्र में
        val eyeMidX = (leftEye.x + rightEye.x) * 0.5f
        val eyeMidY = (leftEye.y + rightEye.y) * 0.5f
        val mouthMidX = (mouthLeft.x + mouthRight.x) * 0.5f
        val mouthMidY = (mouthLeft.y + mouthRight.y) * 0.5f
        val eyeDist = hypot((rightEye.x - leftEye.x).toDouble(), (rightEye.y - leftEye.y).toDouble()).toFloat().coerceAtLeast(12f)

        val noseOffsetRatio = abs(noseBase.x - eyeMidX) / eyeDist
        val mouthOffsetRatio = abs(mouthMidX - noseBase.x) / eyeDist
        val verticalOrderOk = (noseBase.y > eyeMidY + box.height() * 0.08f) &&
                (mouthMidY > noseBase.y + box.height() * 0.08f)

        val nosePerfect = verticalOrderOk && (noseOffsetRatio <= 0.13f) && (mouthOffsetRatio <= 0.15f)
        val noseScore = when {
            nosePerfect -> 0.20f
            verticalOrderOk && noseOffsetRatio <= 0.24f -> 0.11f
            else -> 0.03f
        }

        // 5. दोनों कान (LEFT_EAR & RIGHT_EAR) दिखने और बराबर अनुपात में होने की सख्त जाँच
        val leftEar: PointF? = face.getLandmark(FaceLandmark.LEFT_EAR)?.position
        val rightEar: PointF? = face.getLandmark(FaceLandmark.RIGHT_EAR)?.position

        var earsPerfect = false
        var earScore = 0.0f
        if (leftEar != null && rightEar != null) {
            val minEarX = min(leftEar.x, rightEar.x)
            val maxEarX = max(leftEar.x, rightEar.x)
            val minEyeX = min(leftEye.x, rightEye.x)
            val maxEyeX = max(leftEye.x, rightEye.x)

            // दोनों कान आँखों के बाहरी किनारों से बाहर साफ दिखने चाहिए
            val earsOutsideEyes = (minEarX < minEyeX - eyeDist * 0.08f) && (maxEarX > maxEyeX + eyeDist * 0.08f)
            val distLeftEarToNose = noseBase.x - minEarX
            val distRightEarToNose = maxEarX - noseBase.x

            if (earsOutsideEyes && distLeftEarToNose > 4f && distRightEarToNose > 4f) {
                val earSymmetryRatio = min(distLeftEarToNose, distRightEarToNose) / max(distLeftEarToNose, distRightEarToNose)
                if (earSymmetryRatio >= STRICT_KYC_MIN_EAR_SYMMETRY) {
                    earsPerfect = true
                    earScore = 0.20f
                } else {
                    earScore = (earSymmetryRatio * 0.15f).coerceIn(0.04f, 0.14f)
                }
            } else {
                earScore = 0.05f
            }
        } else if (leftEar != null || rightEar != null) {
            // केवल एक कान दिख रहा है -> अधूरा
            earScore = 0.06f
        }

        // 6. सख्त 5.0° कोण जाँच (Yaw Y, Roll Z, Pitch X सभी <= 5.0°)
        val yaw = abs(face.headEulerAngleY)
        val roll = abs(face.headEulerAngleZ)
        val pitch = abs(face.headEulerAngleX)

        val anglesPerfect = (yaw <= STRICT_KYC_MAX_ANGLE_DEG) &&
                (roll <= STRICT_KYC_MAX_ANGLE_DEG) &&
                (pitch <= STRICT_KYC_MAX_ANGLE_DEG)

        val maxAngle = max(yaw, max(roll, pitch))
        val angleScore = when {
            anglesPerfect -> 0.20f
            maxAngle <= 9.0f -> 0.12f
            maxAngle <= 15.0f -> 0.06f
            else -> 0.01f
        }

        val allStrictConditionsMet = sizeOk && centerOk && eyesPerfect && nosePerfect && earsPerfect && anglesPerfect
        val rawTotal = (sizeAndCenterScore + eyeScore + noseScore + earScore + angleScore).coerceIn(0f, 1f)

        // जब तक सारी शर्तें 100% पूरी न हों, सर्कल 84% से ऊपर कभी न भरे!
        val finalFraction = if (allStrictConditionsMet) {
            1.0f
        } else {
            min(rawTotal, 0.84f)
        }

        return StrictKycProgress(finalFraction, allStrictConditionsMet)
    }

    fun validateKycFaceQuality(face: Face, frameBmp: Bitmap): Pair<Boolean, String> {
        val status = evaluateStrictKycProgress(face, frameBmp.width, frameBmp.height)
        return Pair(status.isPerfectFrame, "")
    }

    // ★ पासपोर्ट/आईडी कार्ड जैसी सुंदर 1:1 DP क्रॉप (जिसमें सिर के ऊपर स्पेस और नीचे दोनों कंधे साफ और बराबर दिखें!):
    fun cropTrueSquarePortraitFromFrames(rawSource: Bitmap, detectBmp: Bitmap, face: Face): Bitmap {
        val scaleX = rawSource.width.toFloat() / detectBmp.width.toFloat().coerceAtLeast(1f)
        val scaleY = rawSource.height.toFloat() / detectBmp.height.toFloat().coerceAtLeast(1f)
        val box = face.boundingBox

        val cx = (box.exactCenterX() * scaleX).roundToInt()
        // केंद्र को +0.20f नीचे (गर्दन की ओर) रखें और साइज़ 2.25f रखें ताकि दोनों कंधे (Shoulders) और कॉलर साफ दिखें
        val cy = ((box.exactCenterY() + (box.height() * 0.20f)) * scaleY).roundToInt()
        val faceW = box.width() * scaleX
        val faceH = box.height() * scaleY

        val desiredSide = (max(faceW, faceH) * 2.25f).roundToInt()
        val maxPossibleSide = min(rawSource.width, rawSource.height)
        val side = desiredSide.coerceIn(120, maxPossibleSide)
        val half = side / 2

        var left = cx - half
        var top = cy - half

        if (left < 0) left = 0
        if (top < 0) top = 0
        if (left + side > rawSource.width) left = max(0, rawSource.width - side)
        if (top + side > rawSource.height) top = max(0, rawSource.height - side)

        val actualW = min(side, rawSource.width - left)
        val actualH = min(side, rawSource.height - top)
        val finalSquareSide = min(actualW, actualH).coerceAtLeast(1)

        val squareCrop = Bitmap.createBitmap(rawSource, left, top, finalSquareSide, finalSquareSide)
        return Bitmap.createScaledBitmap(squareCrop, 400, 400, true)
    }

    fun cropTrueSquarePortrait(source: Bitmap, face: Face): Bitmap {
        return cropTrueSquarePortraitFromFrames(source, source, face)
    }

    // ★ अपग्रेडेड AI स्टूडियो ब्यूटी पोर्ट्रेट फिल्टर (Auto White-Balance + Shadow Fill-Light + Bilateral Skin Smoothing + Crisp Eyes):
    fun applyOnDeviceStudioPortraitFilter(squareBmp: Bitmap): Bitmap {
        val w = squareBmp.width
        val h = squareBmp.height
        if (w < 16 || h < 16) return squareBmp

        val inPixels = IntArray(w * h)
        squareBmp.getPixels(inPixels, 0, w, 0, 0, w, h)

        // 1. चेहरे के मध्य भाग (Center 60%) से औसत ब्राइटनेस और RGB संतुलन (Auto White Balance) निकालें
        var sumR = 0L
        var sumG = 0L
        var sumB = 0L
        var sumLum = 0L
        var count = 0
        val xStart = (w * 0.20f).toInt()
        val xEnd = (w * 0.80f).toInt()
        val yStart = (h * 0.15f).toInt()
        val yEnd = (h * 0.75f).toInt()

        for (y in yStart until yEnd) {
            val rowOffset = y * w
            for (x in xStart until xEnd) {
                val px = inPixels[rowOffset + x]
                val r = (px shr 16) and 0xFF
                val g = (px shr 8) and 0xFF
                val b = px and 0xFF
                val lum = (r * 77 + g * 150 + b * 29) shr 8
                if (lum in 18..242) {
                    sumR += r
                    sumG += g
                    sumB += b
                    sumLum += lum
                    count++
                }
            }
        }

        val safeCount = max(count, 1)
        val avgR = (sumR / safeCount).toFloat().coerceAtLeast(25f)
        val avgG = (sumG / safeCount).toFloat().coerceAtLeast(25f)
        val avgB = (sumB / safeCount).toFloat().coerceAtLeast(25f)
        val avgLum = (sumLum / safeCount).toFloat().coerceIn(30f, 220f)

        // कमरे की नीली या पीली रोशनी को हटाकर प्राकृतिक वार्म स्किन टोन देना
        val grayTarget = (avgR + avgG + avgB) / 3.0f
        val gainR = ((grayTarget / avgR) * 1.035f).coerceIn(0.88f, 1.24f)
        val gainG = ((grayTarget / avgG) * 1.005f).coerceIn(0.90f, 1.18f)
        val gainB = ((grayTarget / avgB) * 0.965f).coerceIn(0.82f, 1.18f)

        // अंधेरे चेहरे को चमकाने के लिए स्मार्ट स्टूडियो गामा और शैडो लिफ्ट
        val targetGamma = when {
            avgLum < 75f -> 0.64
            avgLum < 105f -> 0.74
            avgLum < 135f -> 0.84
            avgLum > 185f -> 1.04
            else -> 0.92
        }
        val shadowLift = if (avgLum < 120f) ((120f - avgLum) * 0.32f).coerceIn(0f, 24f) else 0f

        val toneLut = IntArray(256)
        for (i in 0..255) {
            val norm = (i / 255.0).coerceIn(0.0, 1.0)
            // सॉफ्ट शैडो फिल + गामा ब्राइटनिंग
            val gammaVal = norm.pow(targetGamma)
            // हल्का S-curve ताकि चेहरा धुंधला न होकर चमकदार और स्पष्ट दिखे
            val contrastVal = gammaVal * gammaVal * (3.0 - 2.0 * gammaVal)
            val blended = (gammaVal * 0.68 + contrastVal * 0.32) * 255.0 + shadowLift * (1.0 - norm)
            toneLut[i] = blended.roundToInt().coerceIn(0, 255)
        }

        // 2. टोन मैपिंग + वाइब्रेंस (Natural Portrait Vibrance)
        val tonedPixels = IntArray(w * h)
        for (i in inPixels.indices) {
            val px = inPixels[i]
            val r0 = ((px shr 16) and 0xFF) * gainR
            val g0 = ((px shr 8) and 0xFF) * gainG
            val b0 = (px and 0xFF) * gainB

            var r = toneLut[r0.roundToInt().coerceIn(0, 255)]
            var g = toneLut[g0.roundToInt().coerceIn(0, 255)]
            var b = toneLut[b0.roundToInt().coerceIn(0, 255)]

            // स्मार्ट वाइब्रेंस: फीके रंगों को प्राकृतिक चमक दें
            val mean = (r + g + b) / 3
            val maxC = max(r, max(g, b))
            val minC = min(r, min(g, b))
            val sat = maxC - minC
            val vibFactor = if (sat < 85) 1.14f else 1.05f

            r = (mean + (r - mean) * vibFactor).roundToInt().coerceIn(0, 255)
            g = (mean + (g - mean) * vibFactor).roundToInt().coerceIn(0, 255)
            b = (mean + (b - mean) * vibFactor).roundToInt().coerceIn(0, 255)

            tonedPixels[i] = (0xFF shl 24) or (r shl 16) or (g shl 8) or b
        }

        // 3. Edge-Preserving Bilateral Skin Smoothing + Eye/Collar Detail Sharpening:
        // यह त्वचा के दाग/ग्रेन को मुलायम (Smooth) करता है और आँखों, भौंहों व बालों को एकदम शार्प रखता है
        val finalPixels = IntArray(w * h)
        val colorThreshold = 26

        for (y in 0 until h) {
            val row = y * w
            for (x in 0 until w) {
                val idx = row + x
                if (x < 2 || x >= w - 2 || y < 2 || y >= h - 2) {
                    finalPixels[idx] = tonedPixels[idx]
                    continue
                }

                val centerPx = tonedPixels[idx]
                val cr = (centerPx shr 16) and 0xFF
                val cg = (centerPx shr 8) and 0xFF
                val cb = centerPx and 0xFF

                var wSum = 4
                var rAcc = cr * 4
                var gAcc = cg * 4
                var bAcc = cb * 4

                // 5x5 स्पैशियल-रेंज सैंपलिंग
                val offsetsY = intArrayOf(-2, -1, 0, 1, 2)
                val offsetsX = intArrayOf(-2, -1, 0, 1, 2)
                for (dy in offsetsY) {
                    val ny = (y + dy) * w
                    for (dx in offsetsX) {
                        if (dx == 0 && dy == 0) continue
                        val nPx = tonedPixels[ny + x + dx]
                        val nr = (nPx shr 16) and 0xFF
                        val ng = (nPx shr 8) and 0xFF
                        val nb = nPx and 0xFF

                        val diff = abs(cr - nr) + abs(cg - ng) + abs(cb - nb)
                        if (diff < colorThreshold * 3) {
                            val weight = if (diff < colorThreshold) 2 else 1
                            rAcc += nr * weight
                            gAcc += ng * weight
                            bAcc += nb * weight
                            wSum += weight
                        }
                    }
                }

                val smoothR = rAcc / wSum
                val smoothG = gAcc / wSum
                val smoothB = bAcc / wSum

                // यदि यह किनारा (आँख/बाल/कॉलर) है तो हल्का Unsharp Mask लगाएं, और यदि त्वचा है तो मुलायम रखें
                val edgeDiff = abs(cr - smoothR) + abs(cg - smoothG) + abs(cb - smoothB)
                val outR: Int
                val outG: Int
                val outB: Int
                if (edgeDiff > 18) {
                    outR = (cr + (cr - smoothR) * 0.35f).roundToInt().coerceIn(0, 255)
                    outG = (cg + (cg - smoothG) * 0.35f).roundToInt().coerceIn(0, 255)
                    outB = (cb + (cb - smoothB) * 0.35f).roundToInt().coerceIn(0, 255)
                } else {
                    outR = (smoothR * 0.78f + cr * 0.22f).roundToInt().coerceIn(0, 255)
                    outG = (smoothG * 0.78f + cg * 0.22f).roundToInt().coerceIn(0, 255)
                    outB = (smoothB * 0.78f + cb * 0.22f).roundToInt().coerceIn(0, 255)
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

    // ★ सभी 10+ Gemini Keys पर रोटेशन + हाई-क्वालिटी ऑन-डिवाइस स्टूडियो ब्यूटी फॉलबैक:
    fun enhancePortraitWithGeminiOrFallbackBlocking(context: Context, squareBaseBmp: Bitmap): Pair<Bitmap, Boolean> {
        var fallbackReason = "Gemini Key उपलब्ध नहीं है — ऑन-डिवाइस AI स्टूडियो ब्यूटी फिल्टर लगाया गया"
        try {
            val vm = RealisticVoiceManager(context)
            val keyPool = vm.getGeminiKeysListPublic()

            if (keyPool.isNotEmpty()) {
                val inputBaos = ByteArrayOutputStream()
                Bitmap.createScaledBitmap(squareBaseBmp, 400, 400, true)
                    .compress(Bitmap.CompressFormat.JPEG, 92, inputBaos)
                val inputB64 = Base64.encodeToString(inputBaos.toByteArray(), Base64.NO_WRAP)

                val modelsToTry = listOf(
                    "gemini-2.0-flash-exp-image-generation",
                    "gemini-2.5-flash-image-preview"
                )

                // पहली Key फेल होने पर रुकने के बजाय अधिकतम 4 उपलब्ध Keys तक प्रयास करें
                for (apiKey in keyPool.take(4)) {
                    for (modelName in modelsToTry) {
                        try {
                            val endpoint = "https://generativelanguage.googleapis.com/v1beta/models/$modelName:generateContent?key=$apiKey"
                            val conn = (URL(endpoint).openConnection() as HttpURLConnection).apply {
                                requestMethod = "POST"
                                connectTimeout = 8000
                                readTimeout = 14000
                                doOutput = true
                                setRequestProperty("Content-Type", "application/json; charset=UTF-8")
                            }

                            val prompt = "Enhance this passport/ID card portrait photo (showing head and both shoulders) into a clean, bright, professional studio-lit 1:1 square photo. Keep the person's exact facial identity, eyes, nose, ears, hair, and clothing 100% unchanged. Improve lighting, skin clarity, natural warm skin tone, and crispness."

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
                                                        "Gemini Image Model ($modelName) से कंधों सहित सुंदर 1:1 DP बनी"
                                                    )
                                                    return Pair(Bitmap.createScaledBitmap(sq, 400, 400, true), true)
                                                }
                                            }
                                        }
                                    }
                                }
                                fallbackReason = "Gemini ($modelName) ने इमेज नहीं लौटाई — ऑन-डिवाइस स्टूडियो ब्यूटी फिल्टर लगा"
                            } else {
                                fallbackReason = "Gemini Image API HTTP $code — ऑन-डिवाइस स्टूडियो ब्यूटी फिल्टर से 1:1 DP बनी"
                                if (code == 400 || code == 403 || code == 429) break
                            }
                        } catch (e: UnknownHostException) {
                            fallbackReason = "इंटरनेट बंद था — ऑन-डिवाइस स्टूडियो ब्यूटी फिल्टर से 1:1 DP बनाई गई"
                            break
                        } catch (e: SocketTimeoutException) {
                            fallbackReason = "नेटवर्क धीमा (Timeout) था — ऑन-डिवाइस स्टूडियो ब्यूटी फिल्टर से 1:1 DP बनाई गई"
                            break
                        } catch (e: Exception) {
                            fallbackReason = "कनेक्शन त्रुटि (${e.javaClass.simpleName}) — ऑन-डिवाइस स्टूडियो ब्यूटी फिल्टर लगा"
                            break
                        }
                    }
                }
            }
        } catch (e: Exception) {}

        AiDiagnosticLogger.recordEvent(
            context,
            "DP_EDIT",
            "3. AI स्टूडियो DP सुधार (KYC)",
            "🔄 ऑन-डिवाइस स्टूडियो ब्यूटी",
            fallbackReason
        )
        return Pair(applyOnDeviceStudioPortraitFilter(squareBaseBmp), false)
    }

    // फुल-एचडी और 480p दोनों फ्रेम्स से सटीक 112x112 बायोमेट्रिक चेहरा अलाइन करना
    fun alignAndCropFaceBitmapFromFrames(rawSource: Bitmap, detectBmp: Bitmap, face: Face): Bitmap {
        val scaleX = rawSource.width.toFloat() / detectBmp.width.toFloat().coerceAtLeast(1f)
        val scaleY = rawSource.height.toFloat() / detectBmp.height.toFloat().coerceAtLeast(1f)
        val box = face.boundingBox
        val cx = box.exactCenterX() * scaleX
        val cy = box.exactCenterY() * scaleY
        val cropSide = (max(box.width() * scaleX, box.height() * scaleY) * 0.96f).coerceAtLeast(48f)

        var angleDeg = 0f
        val leftEye = face.getLandmark(FaceLandmark.LEFT_EYE)?.position
        val rightEye = face.getLandmark(FaceLandmark.RIGHT_EYE)?.position
        if (leftEye != null && rightEye != null) {
            val dy = ((rightEye.y - leftEye.y) * scaleY).toDouble()
            val dx = ((rightEye.x - leftEye.x) * scaleX).toDouble()
            val rawAngle = Math.toDegrees(atan2(dy, dx)).toFloat()
            if (abs(rawAngle) in 1.0f..26.0f) {
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
        canvas.drawBitmap(rawSource, matrix, paint)
        return outBmp
    }

    fun alignAndCropFaceBitmap(source: Bitmap, face: Face): Bitmap {
        return alignAndCropFaceBitmapFromFrames(source, source, face)
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

    private fun flipHorizontal(bmp: Bitmap): Bitmap {
        val m = Matrix().apply { preScale(-1f, 1f) }
        return Bitmap.createBitmap(bmp, 0, 0, bmp.width, bmp.height, m, true)
    }

    fun extractHighPrecisionKycEmbeddingList(alignedFaceBmp: Bitmap, hdAlignedFaceBmp: Bitmap? = null): List<Float> {
        val v1 = computeEmbedding(alignedFaceBmp) ?: return emptyList()
        val v2 = try { computeEmbedding(flipHorizontal(alignedFaceBmp)) } catch (e: Exception) { null }
        val v3 = if (hdAlignedFaceBmp != null) computeEmbedding(hdAlignedFaceBmp) else null
        val v4 = if (hdAlignedFaceBmp != null) {
            try { computeEmbedding(flipHorizontal(hdAlignedFaceBmp)) } catch (e: Exception) { null }
        } else null

        val list = listOfNotNull(v1, v2, v3, v4)
        val merged = averageEmbeddings(list)
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
        if (student.embeddings.isEmpty()) return null
        val vectors = mutableListOf<FloatArray>()

        for (savedList in student.embeddings) {
            if (savedList.size == embeddingDim) {
                val arr = FloatArray(embeddingDim)
                for (i in 0 until embeddingDim) {
                    arr[i] = savedList[i]
                }
                vectors.add(l2Normalize(arr))
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
            if (stu.embeddings.isEmpty()) continue
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
        val requiredMarginGap = if (isManualTap) 0.034f else MIN_COSINE_MARGIN_GAP
        val requiredPopGap = if (isManualTap) 0.052f else MIN_POPULATION_GAP

        val isHighConf = (bestCosine >= requiredMinCos) &&
                (isSingleStudent || bestCosine >= 0.77f || (cosineGap >= requiredMarginGap && populationGap >= requiredPopGap))

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
