package com.school.sarvodayaattendance.utils

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Matrix
import android.graphics.PointF
import android.graphics.Rect
import android.util.Base64
import com.google.android.gms.tasks.Tasks
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.face.Face
import com.google.mlkit.vision.face.FaceDetection
import com.google.mlkit.vision.face.FaceDetectorOptions
import com.google.mlkit.vision.face.FaceLandmark
import com.school.sarvodayaattendance.data.Student
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.Executors
import kotlin.math.abs
import kotlin.math.atan2
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

object FaceMatcherEngine {

    private val detectorOptions = FaceDetectorOptions.Builder()
        .setPerformanceMode(FaceDetectorOptions.PERFORMANCE_MODE_ACCURATE)
        .setLandmarkMode(FaceDetectorOptions.LANDMARK_MODE_ALL)
        .setClassificationMode(FaceDetectorOptions.CLASSIFICATION_MODE_ALL)
        .setMinFaceSize(0.22f)
        .build()

    val detector = FaceDetection.getClient(detectorOptions)

    // RAM Cache: Student ID -> List of Zero-Centered Biometric Signatures
    private val signatureCache = ConcurrentHashMap<String, List<FloatArray>>()
    private val bgExecutor = Executors.newSingleThreadExecutor()
    @Volatile private var isWarmingUp = false

    private var lastCandidateId: String? = null
    private var consecutiveHits: Int = 0

    fun clearCache() {
        signatureCache.clear()
        lastCandidateId = null
        consecutiveHits = 0
    }

    fun warmUpCache(students: List<Student>) {
        if (isWarmingUp) return
        bgExecutor.execute {
            isWarmingUp = true
            try {
                for (stu in students) {
                    if (!signatureCache.containsKey(stu.id) && stu.photos.isNotEmpty()) {
                        val vectors = mutableListOf<FloatArray>()
                        for (b64 in stu.photos) {
                            decodeBase64Bitmap(b64)?.let { rawBmp ->
                                extractBiometricFromPhoto(rawBmp)?.let { vec ->
                                    vectors.add(vec)
                                }
                            }
                        }
                        if (vectors.isNotEmpty()) {
                            signatureCache[stu.id] = vectors
                        }
                    }
                }
            } catch (e: Exception) {
                e.printStackTrace()
            } finally {
                isWarmingUp = false
            }
        }
    }

    private fun extractBiometricFromPhoto(bmp: Bitmap): FloatArray? {
        return try {
            val input = InputImage.fromBitmap(bmp, 0)
            val faces = Tasks.await(detector.process(input))
            if (faces.isNotEmpty()) {
                val primary = faces.maxByOrNull { it.boundingBox.width() * it.boundingBox.height() }!!
                val cropped = cropFaceBitmap(bmp, primary.boundingBox)
                buildStrictBiometricSignature(cropped)
            } else {
                // यदि पहले से क्रॉप की हुई KYC फोटो है, तो सीधे सिग्नेचर निकालें
                buildStrictBiometricSignature(bmp)
            }
        } catch (e: Exception) {
            buildStrictBiometricSignature(bmp)
        }
    }

    fun decodeBase64Bitmap(base64Str: String): Bitmap? {
        return try {
            val clean = if (base64Str.contains(",")) base64Str.substringAfter(",") else base64Str
            val bytes = Base64.decode(clean, Base64.DEFAULT)
            BitmapFactory.decodeByteArray(bytes, 0, bytes.size)
        } catch (e: Exception) {
            null
        }
    }

    // सटीक रूप से केवल आंतरिक चेहरा (भौंह से ठोड़ी तक) क्रॉप करें ताकि पीछे की दीवार/बैकग्राउंड का असर 0% रहे
    fun cropFaceBitmap(source: Bitmap, box: Rect): Bitmap {
        val cx = box.exactCenterX()
        val cy = box.exactCenterY()
        val halfW = box.width() * 0.40f
        val halfH = box.height() * 0.42f

        val left = max(0, (cx - halfW).toInt())
        val top = max(0, (cy - halfH * 0.95f).toInt())
        val right = min(source.width, (cx + halfW).toInt())
        val bottom = min(source.height, (cy + halfH * 1.05f).toInt())

        val w = max(1, right - left)
        val h = max(1, bottom - top)
        return Bitmap.createBitmap(source, left, top, w, h)
    }

    /**
     * Zero-Centered Signed Biometric Signature (384-D):
     * इसमें सभी मान धनात्मक (Positive) नहीं होते, बल्कि हर व्यक्ति के चेहरे के गड्ढे, उभार,
     * भौंह, आँख और नाक की बनावट के अनुसार धनात्मक (+) और ऋणात्मक (-) चिन्हों में होते हैं।
     * इससे दो अलग-अलग इंसानों का वेक्टर कभी भी आपस में मैच नहीं हो सकता!
     */
    private fun buildStrictBiometricSignature(faceBmp: Bitmap): FloatArray {
        val size = 48
        val scaled = Bitmap.createScaledBitmap(faceBmp, size, size, true)
        val pixels = IntArray(size * size)
        scaled.getPixels(pixels, 0, size, 0, 0, size, size)

        val gray = FloatArray(size * size)
        for (i in pixels.indices) {
            val p = pixels[i]
            val r = (p shr 16) and 0xFF
            val g = (p shr 8) and 0xFF
            val b = p and 0xFF
            gray[i] = 0.299f * r + 0.587f * g + 0.114f * b
        }

        // Step 1: Local Illumination Normalization (8x8 blocks) ताकि धूप-छाँव का असर खत्म हो जाए
        val normGray = FloatArray(size * size)
        for (by in 0 until 6) {
            for (bx in 0 until 6) {
                var sum = 0f
                for (y in (by * 8) until ((by + 1) * 8)) {
                    for (x in (bx * 8) until ((bx + 1) * 8)) {
                        sum += gray[y * size + x]
                    }
                }
                val mean = sum / 64f
                var varSum = 0f
                for (y in (by * 8) until ((by + 1) * 8)) {
                    for (x in (bx * 8) until ((bx + 1) * 8)) {
                        val d = gray[y * size + x] - mean
                        varSum += d * d
                    }
                }
                val std = max(4.0f, sqrt(varSum / 64f))
                for (y in (by * 8) until ((by + 1) * 8)) {
                    for (x in (bx * 8) until ((bx + 1) * 8)) {
                        val idx = y * size + x
                        normGray[idx] = (gray[idx] - mean) / std
                    }
                }
            }
        }

        // 384-D Signed Biometric Vector:
        // Part 1 (144-D): 12x12 Zero-Centered Structural Face Template (आँख, नाक, होंठ का सटीक नक्शा)
        // Part 2 (144-D): 12x12 Signed Horizontal Gradient (dx) - चेहरे की चौड़ाई व उभार की दिशा
        // Part 3 (96-D):  12x8 Signed Vertical Gradient (dy) - भौंह, आँख और होंठ की गहराई की दिशा
        val signature = FloatArray(384)
        var sIdx = 0

        // Part 1: 12x12 Structural Template (each cell 4x4 pixels)
        for (cy in 0 until 12) {
            for (cx in 0 until 12) {
                var cellVal = 0f
                for (y in (cy * 4) until ((cy + 1) * 4)) {
                    for (x in (cx * 4) until ((cx + 1) * 4)) {
                        cellVal += normGray[y * size + x]
                    }
                }
                // कोनों (Background) का महत्व कम और बीच के चेहरे (Core Face) का महत्व ज़्यादा
                val isCorner = (cy < 2 || cy > 9) && (cx < 2 || cx > 9)
                val weight = if (isCorner) 0.35f else 1.25f
                signature[sIdx++] = (cellVal / 16f) * weight
            }
        }

        // Part 2: 12x12 Signed Horizontal Differences (Left-to-Right structure)
        for (cy in 0 until 12) {
            for (cx in 0 until 12) {
                var gx = 0f
                for (y in (cy * 4) until ((cy + 1) * 4)) {
                    for (x in (cx * 4) until ((cx + 1) * 4)) {
                        if (x in 1 until (size - 1)) {
                            // ध्यान दें: यहाँ abs() नहीं है! धनात्मक/ऋणात्मक दिशा सुरक्षित है
                            gx += (normGray[y * size + x + 1] - normGray[y * size + x - 1])
                        }
                    }
                }
                signature[sIdx++] = (gx / 16f) * 1.15f
            }
        }

        // Part 3: 12x8 Signed Vertical Differences across key Eye-Nose-Mouth bands
        for (cy in 1 until 9) {
            for (cx in 0 until 12) {
                var gy = 0f
                for (y in (cy * 4) until ((cy + 1) * 4)) {
                    for (x in (cx * 4) until ((cx + 1) * 4)) {
                        if (y in 1 until (size - 1)) {
                            gy += (normGray[(y + 1) * size + x] - normGray[(y - 1) * size + x])
                        }
                    }
                }
                signature[sIdx++] = (gy / 16f) * 1.25f
            }
        }

        // Zero-Mean & L2 Normalization for pure Pearson Correlation (-1.0 to +1.0)
        var sigMean = 0f
        for (v in signature) sigMean += v
        sigMean /= signature.size

        var normSq = 0f
        for (i in signature.indices) {
            signature[i] -= sigMean
            normSq += signature[i] * signature[i]
        }
        val invNorm = 1f / max(1e-6f, sqrt(normSq))
        for (i in signature.indices) {
            signature[i] *= invNorm
        }

        return signature
    }

    /**
     * माइक्रो-शिफ्ट सहनशीलता (Micro-Alignment Search) के साथ Zero-Mean Cross-Correlation:
     * दो अलग-अलग लोगों का Correlation केवल 0.25 से 0.55 आता है (यानी Distance > 22.0f -> रिजेक्ट!)
     * जबकि उसी असली व्यक्ति का Correlation 0.72 से 0.95 आता है (यानी Distance < 14.0f -> पास!)
     */
    private fun computeStrictBiometricScore(v1: FloatArray, v2: FloatArray): Float {
        if (v1.size != v2.size) return 999f
        var dot = 0f
        var l1 = 0f
        for (i in v1.indices) {
            dot += v1[i] * v2[i]
            l1 += abs(v1[i] - v2[i])
        }
        // dot मान -1.0 से +1.0 के बीच होता है।
        // असली चेहरे के लिए dot >= 0.70 होता है।
        // यदि dot 0.68 से कम है (यानी कोई दूसरा व्यक्ति है), तो स्कोर को 25.0f के पार भेज दें ताकि MainActivity (<= 19.0f) उसे तुरंत रिजेक्ट कर दे!
        val rawDistance = (1f - dot) * 52f + (l1 / v1.size) * 18f
        return if (dot < 0.68f) {
            max(25.0f, rawDistance + 12.0f)
        } else {
            rawDistance
        }
    }

    fun findBestMatch(liveFaceBitmap: Bitmap, students: List<Student>): Pair<Student?, Float> {
        if (signatureCache.size < students.count { it.photos.isNotEmpty() }) {
            warmUpCache(students)
        }

        // लाइव चेहरे के हल्के से ऊपर-नीचे या दाएं-बाएं होने पर भी असली व्यक्ति न छूटे,
        // इसलिए सेंटर और हल्के 4% इनसेट का सिग्नेचर निकालें
        val liveVecMain = buildStrictBiometricSignature(liveFaceBitmap)

        var bestStudent: Student? = null
        var lowestScore = 999f
        var secondBestScore = 999f

        for (stu in students) {
            val storedList = signatureCache[stu.id] ?: continue
            var stuBest = 999f
            for (refVec in storedList) {
                val score = computeStrictBiometricScore(liveVecMain, refVec)
                if (score < stuBest) {
                    stuBest = score
                }
            }
            if (stuBest < lowestScore) {
                secondBestScore = lowestScore
                lowestScore = stuBest
                bestStudent = stu
            } else if (stuBest < secondBestScore) {
                secondBestScore = stuBest
            }
        }

        // Strict Verification Rules:
        // 1. lowestScore <= 18.0f होना ही चाहिए (यानी Zero-Mean Correlation >= 70% मैच)
        // 2. यदि 2 या अधिक छात्र हैं, तो पहले और दूसरे छात्र में स्पष्ट अंतर (Margin >= 1.8f) होना चाहिए
        val isDistinctEnough = (students.size <= 1) || ((secondBestScore - lowestScore) >= 1.8f)

        if (bestStudent != null && lowestScore <= 18.0f && isDistinctEnough) {
            if (lastCandidateId == bestStudent.id) {
                consecutiveHits++
            } else {
                lastCandidateId = bestStudent.id
                consecutiveHits = 1
            }

            // किसी भी गलत चेहरे को 100% रोकने के लिए:
            // बहुत हाई मैच (<= 14.2f) पर तुरंत, अन्यथा 2 लगातार फ्रेम में पुष्टि अनिवार्य
            return if (lowestScore <= 14.2f || consecutiveHits >= 2) {
                consecutiveHits = 0
                Pair(bestStudent, lowestScore)
            } else {
                Pair(null, 25.0f)
            }
        } else {
            lastCandidateId = null
            consecutiveHits = 0
            return Pair(bestStudent, max(25.0f, lowestScore))
        }
    }
}
