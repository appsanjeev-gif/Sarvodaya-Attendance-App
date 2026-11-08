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
import com.google.mlkit.vision.common.InputImage
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
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
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
        val isVerified192D: Boolean,
        val poseEmbeddings: List<FloatArray>,
        val centroidEmbedding: FloatArray
    )

    private const val VERIFIED_KYC_PREF = "SarvodayaVerified192DKycV8"
    private const val KEY_VERIFIED_STU_IDS = "verified_192d_student_ids"
    private const val KEY_VERIFIED_EMB_SIGS = "verified_192d_emb_sigs"
    private const val KEY_V8_INIT_CLEANED = "v8_polluted_emb_cleaned"
    private const val OLD_PHOTO_VEC_CACHE_PREF = "SarvodayaOldPhotoVecCacheV8"

    @Volatile
    private var tfliteInterpreter: Interpreter? = null
    private var embeddingDim: Int = 192
    private var inputImageSize: Int = 112
    private val templateCache = ConcurrentHashMap<String, StudentBiometricTemplate>()

    private val verifiedStudentIds = ConcurrentHashMap.newKeySet<String>()
    private val verifiedEmbeddingSigs = ConcurrentHashMap.newKeySet<String>()

    private val isWarmingUp = AtomicBoolean(false)

    @Volatile
    private var initialWarmUpDone = false

    // 1. सत्यापित 192-D छात्रों (Verified 192-D) के लिए सख्त थ्रेशोल्ड (सीधी उपस्थिति):
    const val INSTANT_PASS_COSINE = 0.74f
    const val PASS_VERIFIED_COSINE = 0.675f
    const val MANUAL_VERIFIED_COSINE = 0.645f
    const val MIN_VERIFIED_MARGIN_GAP = 0.042f
    const val MIN_VERIFIED_POP_GAP = 0.065f

    // 2. पुराने 128-D KYC छात्रों (जिनका अभी नया 192-D वेक्टर पक्का नहीं हुआ है) के लिए थ्रेशोल्ड:
    // (यह सीधे हाज़िरी नहीं लगाता, बल्कि 3.4 सेकंड वाला '✔ पक्का करें (OK)' पुष्टि पॉप-अप दिखाता है)
    const val PASS_OLD_KYC_AUTO_COSINE = 0.545f
    const val PASS_OLD_KYC_MANUAL_COSINE = 0.450f
    const val MIN_OLD_KYC_MARGIN_GAP = 0.022f
    const val MIN_OLD_KYC_POP_GAP = 0.038f
    const val AI_ASSIST_COSINE_MIN = 0.50f

    // लाइव कैमरे के लिए सुपर-फास्ट डिटेक्टर
    val detector: FaceDetector by lazy {
        val options = FaceDetectorOptions.Builder()
            .setPerformanceMode(FaceDetectorOptions.PERFORMANCE_MODE_FAST)
            .setLandmarkMode(FaceDetectorOptions.LANDMARK_MODE_ALL)
            .setClassificationMode(FaceDetectorOptions.CLASSIFICATION_MODE_ALL)
            .setMinFaceSize(0.16f)
            .build()
        FaceDetection.getClient(options)
    }

    // पुरानी सेव की गई प्रोफाइल फोटो (stu.photos) में सटीक चेहरा और आँखें ढूँढने के लिए अलग बैकग्राउंड डिटेक्टर
    private val offlinePhotoDetector: FaceDetector by lazy {
        val options = FaceDetectorOptions.Builder()
            .setPerformanceMode(FaceDetectorOptions.PERFORMANCE_MODE_ACCURATE)
            .setLandmarkMode(FaceDetectorOptions.LANDMARK_MODE_ALL)
            .setMinFaceSize(0.12f)
            .build()
        FaceDetection.getClient(options)
    }

    fun isNeuralModelReady(): Boolean = (tfliteInterpreter != null)

    fun isReadyForLiveScan(): Boolean = (tfliteInterpreter != null && initialWarmUpDone)

    fun getEmbeddingDimension(): Int = embeddingDim

    private fun computeEmbeddingSig(emb: List<Float>): String {
        if (emb.size < 10) return ""
        val sb = StringBuilder()
        val step = max(1, emb.size / 12)
        var i = 0
        while (i < emb.size) {
            sb.append(((emb[i] * 1000f).roundToInt())).append("_")
            i += step
        }
        return sb.toString()
    }

    private fun loadVerifiedRegistry(context: Context) {
        try {
            val prefs = context.applicationContext.getSharedPreferences(VERIFIED_KYC_PREF, Context.MODE_PRIVATE)
            prefs.getStringSet(KEY_VERIFIED_STU_IDS, emptySet())?.let { verifiedStudentIds.addAll(it) }
            prefs.getStringSet(KEY_VERIFIED_EMB_SIGS, emptySet())?.let { verifiedEmbeddingSigs.addAll(it) }
        } catch (e: Exception) {}
    }

    private fun saveVerifiedRegistry(context: Context) {
        try {
            val prefs = context.applicationContext.getSharedPreferences(VERIFIED_KYC_PREF, Context.MODE_PRIVATE)
            prefs.edit()
                .putStringSet(KEY_VERIFIED_STU_IDS, HashSet(verifiedStudentIds))
                .putStringSet(KEY_VERIFIED_EMB_SIGS, HashSet(verifiedEmbeddingSigs))
                .apply()
        } catch (e: Exception) {}
    }

    // जब AdminActivity में लाइव KYC फोटो ली जाए, तो उसके 192-D वेक्टर को सत्यापित रजिस्टर में दर्ज करें
    fun registerVerified192DEmbedding(context: Context, emb: List<Float>) {
        if (emb.size != embeddingDim) return
        loadVerifiedRegistry(context)
        val sig = computeEmbeddingSig(emb)
        if (sig.isNotEmpty()) {
            verifiedEmbeddingSigs.add(sig)
            saveVerifiedRegistry(context)
        }
    }

    // जब MainActivity में 3.4s पॉप-अप पर [✔ पक्का करें (OK)] दबाया जाए, तो छात्र को स्थायी रूप से Verified 192-D मार्क करें
    fun markStudentVerified192D(context: Context, studentId: String, emb: List<Float>) {
        loadVerifiedRegistry(context)
        if (studentId.isNotBlank()) {
            verifiedStudentIds.add(studentId)
        }
        if (emb.size == embeddingDim) {
            val sig = computeEmbeddingSig(emb)
            if (sig.isNotEmpty()) verifiedEmbeddingSigs.add(sig)
        }
        saveVerifiedRegistry(context)
    }

    // जांचें कि छात्र के पास मानव-सत्यापित (Human-Verified) 192-D बायोमेट्रिक वेक्टर मौजूद है या नहीं
    fun hasVerified192D(student: Student): Boolean {
        val hasValidDim = student.embeddings.isNotEmpty() && student.embeddings.any { it.size == embeddingDim }
        if (!hasValidDim) return false
        if (verifiedStudentIds.contains(student.id)) return true
        for (emb in student.embeddings) {
            if (emb.size == embeddingDim) {
                val sig = computeEmbeddingSig(emb)
                if (sig.isNotEmpty() && verifiedEmbeddingSigs.contains(sig)) {
                    verifiedStudentIds.add(student.id)
                    return true
                }
            }
        }
        return false
    }

    fun updateSingleStudentCache(student: Student) {
        val built = buildStudentTemplate(student, forceMlKitOnPhoto = false)
        if (built != null) {
            templateCache[student.id] = built
        }
    }

    @Synchronized
    fun initEngine(context: Context) {
        loadVerifiedRegistry(context)
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

    // 2-चरणीय स्मार्ट वार्म-अप:
    // चरण 1 (0.3 सेकंड): तुरंत फास्ट क्रॉप से सभी छात्रों के टेम्पलेट तैयार करें ताकि स्कैनर तुरंत काम करे।
    // चरण 2 (बैकग्राउंड): पुराने बिना-ReKYC वाले छात्रों की फोटो पर अलग offlinePhotoDetector (ML Kit) चलाकर सटीक आँख-अलाइनमेंट करें।
    fun warmUpCache(students: List<Student>, context: Context? = null) {
        if (context != null) {
            loadVerifiedRegistry(context)
        }
        if (students.isEmpty()) {
            initialWarmUpDone = true
            return
        }
        if (isWarmingUp.getAndSet(true)) return

        Thread {
            try {
                val appCtx = context?.applicationContext
                val kycPrefs = appCtx?.getSharedPreferences(VERIFIED_KYC_PREF, Context.MODE_PRIVATE)
                val v8Cleaned = kycPrefs?.getBoolean(KEY_V8_INIT_CLEANED, false) ?: false

                // यदि पिछले V5/V6/V7 बिल्ड ने बिना पूछे सभी छात्रों के stu.embeddings में ऑटो-क्रॉप वेक्टर भर दिए थे,
                // तो एक बार उन असत्यापित वेक्टर्स को साफ करें ताकि प्रत्येक पुराने छात्र के लिए 3.4s वाला Re-KYC पॉप-अप सही से काम करे!
                if (appCtx != null && !v8Cleaned) {
                    var cleanedAny = false
                    for (stu in students) {
                        if (!hasVerified192D(stu) && stu.photos.isNotEmpty() && stu.embeddings.isNotEmpty()) {
                            stu.embeddings.clear()
                            cleanedAny = true
                        }
                    }
                    if (cleanedAny) {
                        try {
                            val masterPayload = AppStorage.loadData(appCtx)
                            for (s in masterPayload.students) {
                                if (!hasVerified192D(s) && s.photos.isNotEmpty()) {
                                    s.embeddings.clear()
                                }
                            }
                            AppStorage.saveData(appCtx, masterPayload)
                        } catch (e: Exception) {}
                    }
                    kycPrefs?.edit()?.putBoolean(KEY_V8_INIT_CLEANED, true)?.apply()
                }

                // किसी भी छात्र के पास यदि नया Verified 192-D सिग्नेचर मैच करता है तो उसकी ID रजिस्टर कर लें
                if (appCtx != null) {
                    var regChanged = false
                    for (stu in students) {
                        if (hasVerified192D(stu) && !verifiedStudentIds.contains(stu.id)) {
                            verifiedStudentIds.add(stu.id)
                            regChanged = true
                        }
                    }
                    if (regChanged) saveVerifiedRegistry(appCtx)
                }

                // चरण 1: तुरंत फास्ट टेम्पलेट तैयार करें (< 0.4 सेकंड)
                for (stu in students) {
                    if (stu.photos.isEmpty() && stu.embeddings.isEmpty()) continue
                    if (!templateCache.containsKey(stu.id)) {
                        val fastBuilt = buildStudentTemplate(stu, forceMlKitOnPhoto = false, context = appCtx)
                        if (fastBuilt != null) {
                            templateCache[stu.id] = fastBuilt
                        }
                    }
                }
                initialWarmUpDone = true

                // चरण 2: पुराने छात्रों की सेव फोटो पर बैकग्राउंड में ML Kit offlinePhotoDetector चलाकर सटीक चेहरा निकालें
                for (stu in students) {
                    if (stu.photos.isEmpty()) continue
                    if (!hasVerified192D(stu)) {
                        val refined = buildStudentTemplate(stu, forceMlKitOnPhoto = true, context = appCtx)
                        if (refined != null) {
                            templateCache[stu.id] = refined
                        }
                    }
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

    // लाइव चेहरे की जांच + स्पष्ट हिंदी कारण
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

        if (abs(face.headEulerAngleY) > 26f || abs(face.headEulerAngleZ) > 24f) {
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

    fun validateKycFaceQuality(face: Face, frameBmp: Bitmap): Pair<Boolean, String> {
        return evaluateLiveFaceGate(face, frameBmp.width, frameBmp.height)
    }

    // ★ KYC DP फोटो क्रॉप फिक्स: यदि डिटेक्शन 480p (detectBmp) पर हुआ है और फोटो फुल-एचडी (rawSource) से काटी जा रही है,
    // तो कोऑर्डिनेट को सटीक अनुपात में स्केल करें ताकि छत/कोने के बजाय 100% चेहरे के बीचों-बीच से साफ 1:1 DP कटे!
    fun cropTrueSquarePortraitFromFrames(rawSource: Bitmap, detectBmp: Bitmap, face: Face): Bitmap {
        val scaleX = rawSource.width.toFloat() / detectBmp.width.toFloat().coerceAtLeast(1f)
        val scaleY = rawSource.height.toFloat() / detectBmp.height.toFloat().coerceAtLeast(1f)
        val box = face.boundingBox

        val cx = (box.exactCenterX() * scaleX).roundToInt()
        val cy = ((box.exactCenterY() - (box.height() * 0.04f)) * scaleY).roundToInt()
        val faceW = box.width() * scaleX
        val faceH = box.height() * scaleY

        val desiredSide = (max(faceW, faceH) * 1.62f).roundToInt()
        val maxPossibleSide = min(rawSource.width, rawSource.height)
        val side = desiredSide.coerceIn(80, maxPossibleSide)
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
        return Bitmap.createScaledBitmap(squareCrop, 320, 320, true)
    }

    fun cropTrueSquarePortrait(source: Bitmap, face: Face): Bitmap {
        return cropTrueSquarePortraitFromFrames(source, source, face)
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

    // पुरानी सेव की गई प्रोफाइल फोटो पर अलग बैकग्राउंड डिटेक्टर चलाकर असली आँख-नाक के अनुसार 112x112 चेहरा निकालना
    private fun detectAndAlignStoredPhotoBlocking(dpBmp: Bitmap): Bitmap? {
        var detectedAligned: Bitmap? = null
        val latch = CountDownLatch(1)
        try {
            val inputImg = InputImage.fromBitmap(dpBmp, 0)
            offlinePhotoDetector.process(inputImg)
                .addOnSuccessListener { faces ->
                    if (faces.isNotEmpty()) {
                        val primary = faces.maxByOrNull { it.boundingBox.width() * it.boundingBox.height() }!!
                        detectedAligned = alignAndCropFaceBitmap(dpBmp, primary)
                    }
                }
                .addOnCompleteListener {
                    latch.countDown()
                }
            latch.await(650, TimeUnit.MILLISECONDS)
        } catch (e: Exception) {}
        return detectedAligned
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

    private fun cosineToPercentage(cos: Float, isVerified: Boolean): Int {
        val passThreshold = if (isVerified) PASS_VERIFIED_COSINE else PASS_OLD_KYC_AUTO_COSINE
        if (cos <= 0.30f) return ((cos.coerceAtLeast(0f) / 0.30f) * 30f).roundToInt()
        if (cos < passThreshold) {
            val norm = (cos - 0.30f) / (passThreshold - 0.30f)
            return (30f + norm * 44f).roundToInt().coerceIn(30, 74)
        }
        val passNorm = ((cos - passThreshold) / (0.88f - passThreshold)).coerceIn(0f, 1f)
        return (76f + passNorm * 23f).roundToInt().coerceIn(76, 99)
    }

    private fun serializeFloatArray(arr: FloatArray): String {
        return arr.joinToString(",") { "%.5f".format(java.util.Locale.US, it) }
    }

    private fun deserializeFloatArray(str: String): FloatArray? {
        if (str.isBlank()) return null
        val parts = str.split(",")
        if (parts.size != embeddingDim) return null
        return try {
            val out = FloatArray(embeddingDim)
            for (i in 0 until embeddingDim) {
                out[i] = parts[i].toFloat()
            }
            l2Normalize(out)
        } catch (e: Exception) {
            null
        }
    }

    private fun buildStudentTemplate(
        student: Student,
        forceMlKitOnPhoto: Boolean = false,
        context: Context? = null
    ): StudentBiometricTemplate? {
        val vectors = mutableListOf<FloatArray>()
        val isVerified = hasVerified192D(student)

        // 1. यदि छात्र के पास सत्यापित 192-D वेक्टर है, तो उसे सर्वोच्च प्राथमिकता दें
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

        // 2. पुरानी सेव की गई प्रोफाइल फोटो (stu.photos) से मल्टी-स्केल और ML-Kit अलाइन्ड वेक्टर बनाएं
        if (student.photos.isNotEmpty()) {
            val p0 = student.photos[0]
            val photoCacheKey = "mlvec_${student.id}_${p0.length}_${p0.takeLast(32).hashCode()}"
            val cachePrefs = context?.applicationContext?.getSharedPreferences(OLD_PHOTO_VEC_CACHE_PREF, Context.MODE_PRIVATE)

            val cachedVecStr = cachePrefs?.getString(photoCacheKey, null)
            if (!cachedVecStr.isNullOrBlank()) {
                deserializeFloatArray(cachedVecStr)?.let { vectors.add(it) }
            } else if (forceMlKitOnPhoto) {
                val bmp0 = decodeBase64Bitmap(p0)
                if (bmp0 != null) {
                    val mlAligned = detectAndAlignStoredPhotoBlocking(bmp0)
                    if (mlAligned != null) {
                        val v1 = computeEmbedding(mlAligned)
                        val v2 = try { computeEmbedding(flipHorizontal(mlAligned)) } catch (e: Exception) { null }
                        val merged = if (v1 != null && v2 != null) averageEmbeddings(listOf(v1, v2)) else v1
                        if (merged != null) {
                            vectors.add(merged)
                            cachePrefs?.edit()?.putString(photoCacheKey, serializeFloatArray(merged))?.apply()
                        }
                    }
                }
            }

            // साथ ही सेव की गई DP के 3 अनुपातिक क्रॉप (0.55f, 0.60f, 0.68f) + मिरर फ्लिप भी रखें ताकि पुराने 128-D के हर ज़ूम के फोटो पकड़ में आएं
            for (b64 in student.photos.take(2)) {
                val bmp = decodeBase64Bitmap(b64) ?: continue
                for (ratio in floatArrayOf(0.593f, 0.66f, 0.53f)) {
                    val crop = extractExactBiometricFaceFromStoredDp(bmp, ratio, 0.022f)
                    val v1 = computeEmbedding(crop)
                    val v2 = try { computeEmbedding(flipHorizontal(crop)) } catch (e: Exception) { null }
                    if (v1 != null) vectors.add(v1)
                    if (v1 != null && v2 != null) {
                        vectors.add(averageEmbeddings(listOf(v1, v2)))
                    }
                }
            }
        }

        if (vectors.isEmpty()) return null
        val centroid = averageEmbeddings(vectors)
        return StudentBiometricTemplate(
            student = student,
            isVerified192D = isVerified,
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

        // 1. शुद्ध रॉ लाइव वेक्टर (Verified 192-D छात्रों के लिए)
        val vRawPrimary = computeEmbedding(liveFaceBmp)
            ?: return MatchResult(null, 999f, 0, 0f, false, false, emptyList())
        val vRawFlipped = try { computeEmbedding(flipHorizontal(liveFaceBmp)) } catch (e: Exception) { null }
        val liveRawMerged = if (vRawFlipped != null) averageEmbeddings(listOf(vRawPrimary, vRawFlipped)) else vRawPrimary

        // 2. स्टूडियो-नॉर्मलाइज्ड लाइव वेक्टर (ताकि पुरानी फिल्टर लगी हुई सेव फोटो stu.photos[0] के साथ 100% समान कंट्रास्ट मिले!)
        val studioLiveBmp = applyOnDeviceStudioPortraitFilter(liveFaceBmp)
        val vStudioPrimary = computeEmbedding(studioLiveBmp) ?: vRawPrimary
        val vStudioFlipped = try { computeEmbedding(flipHorizontal(studioLiveBmp)) } catch (e: Exception) { null }
        val liveStudioMerged = if (vStudioFlipped != null) averageEmbeddings(listOf(vStudioPrimary, vStudioFlipped)) else vStudioPrimary

        data class ScoredCandidate(
            val student: Student,
            val isVerified: Boolean,
            val distance: Float,
            val cosine: Float
        )

        val scoredList = ArrayList<ScoredCandidate>(candidates.size)

        for (stu in candidates) {
            if (stu.photos.isEmpty() && stu.embeddings.isEmpty()) continue
            var template = templateCache[stu.id]
            if (template == null) {
                template = buildStudentTemplate(stu, forceMlKitOnPhoto = false)
                if (template != null) {
                    templateCache[stu.id] = template
                }
            }
            if (template == null) continue

            val isStuVerified = hasVerified192D(stu)
            var maxPoseCos = -1f
            for (poseVec in template.poseEmbeddings) {
                val cRaw = max(
                    calculateCosineSimilarity(vRawPrimary, poseVec),
                    calculateCosineSimilarity(liveRawMerged, poseVec)
                )
                val cMax = if (isStuVerified) {
                    cRaw
                } else {
                    val cStudio = max(
                        calculateCosineSimilarity(vStudioPrimary, poseVec),
                        calculateCosineSimilarity(liveStudioMerged, poseVec)
                    )
                    max(cRaw, cStudio)
                }
                if (cMax > maxPoseCos) maxPoseCos = cMax
            }

            val centroidRaw = calculateCosineSimilarity(liveRawMerged, template.centroidEmbedding)
            val centroidScore = if (isStuVerified) {
                centroidRaw
            } else {
                max(centroidRaw, calculateCosineSimilarity(liveStudioMerged, template.centroidEmbedding))
            }

            val bestScore = (maxPoseCos * 0.86f) + (centroidScore * 0.14f)
            val distance = (1f - bestScore) * 10.0f

            scoredList.add(ScoredCandidate(stu, isStuVerified, distance, bestScore))
        }

        if (scoredList.isEmpty()) {
            return MatchResult(null, 999f, 0, 0f, false, false, emptyList())
        }

        scoredList.sortByDescending { it.cosine }

        val best = scoredList[0]
        val secondBestCosine = if (scoredList.size > 1) scoredList[1].cosine else 0f
        val cosineGap = best.cosine - secondBestCosine

        val otherSlice = scoredList.drop(1).take(5)
        val avgOthers = if (otherSlice.isNotEmpty()) {
            otherSlice.map { it.cosine }.average().toFloat()
        } else {
            0f
        }
        val populationGap = best.cosine - avgOthers
        val bestPct = cosineToPercentage(best.cosine, best.isVerified)

        val topCandidates = scoredList
            .filter { it.cosine >= AI_ASSIST_COSINE_MIN }
            .take(3)
            .map { it.student }

        val isSingleStudent = (scoredList.size == 1)

        val isHighConf = if (best.isVerified) {
            // सत्यापित 192-D छात्र -> सख्त जांच के बाद सीधी हाज़िरी
            val reqCos = if (isManualTap) MANUAL_VERIFIED_COSINE else PASS_VERIFIED_COSINE
            val reqMargin = if (isManualTap) 0.032f else MIN_VERIFIED_MARGIN_GAP
            val reqPop = if (isManualTap) 0.052f else MIN_VERIFIED_POP_GAP
            (best.cosine >= reqCos) &&
                (isSingleStudent || best.cosine >= 0.77f || (cosineGap >= reqMargin && populationGap >= reqPop))
        } else {
            // पुराना 128-D KYC छात्र -> पहचान कर 3.4s वाला '✔ पक्का करें (OK)' री-केवाईसी पॉप-अप दिखाने के लिए
            val reqCos = if (isManualTap) PASS_OLD_KYC_MANUAL_COSINE else PASS_OLD_KYC_AUTO_COSINE
            val reqMargin = if (isManualTap) 0.012f else MIN_OLD_KYC_MARGIN_GAP
            val reqPop = if (isManualTap) 0.022f else MIN_OLD_KYC_POP_GAP
            (best.cosine >= reqCos) &&
                (isSingleStudent || best.cosine >= 0.65f || (cosineGap >= reqMargin && populationGap >= reqPop))
        }

        val needsSecondFrame = !isManualTap && (!best.isVerified || best.cosine < INSTANT_PASS_COSINE)

        return if (isHighConf) {
            MatchResult(best.student, best.distance, bestPct, best.cosine, true, needsSecondFrame, topCandidates)
        } else {
            MatchResult(null, best.distance, bestPct, best.cosine, false, true, topCandidates)
        }
    }

    fun findBestMatch(liveFaceBmp: Bitmap, candidates: List<Student>): Pair<Student?, Float> {
        val detailed = findBestMatchDetailed(liveFaceBmp, candidates, false)
        return Pair(detailed.student, detailed.distance)
    }
}
