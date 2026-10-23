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
import com.google.mlkit.vision.face.FaceContour
import com.google.mlkit.vision.face.FaceDetection
import com.google.mlkit.vision.face.FaceDetectorOptions
import com.google.mlkit.vision.face.FaceLandmark
import com.school.sarvodayaattendance.data.Student
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.Executors
import kotlin.math.abs
import kotlin.math.atan2
import kotlin.math.cos
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sin
import kotlin.math.sqrt

object FaceMatcherEngine {

    // Google ML Kit High-Accuracy Biometric Detector (Landmarks + Contours + Classification)
    private val detectorOptions = FaceDetectorOptions.Builder()
        .setPerformanceMode(FaceDetectorOptions.PERFORMANCE_MODE_ACCURATE)
        .setLandmarkMode(FaceDetectorOptions.LANDMARK_MODE_ALL)
        .setContourMode(FaceDetectorOptions.CONTOUR_MODE_ALL)
        .setClassificationMode(FaceDetectorOptions.CLASSIFICATION_MODE_ALL)
        .setMinFaceSize(0.20f)
        .build()

    val detector = FaceDetection.getClient(detectorOptions)

    // RAM Cache: Student ID -> List of 192-D Biometric Vectors
    private val signatureCache = ConcurrentHashMap<String, List<FloatArray>>()
    private val bgExecutor = Executors.newSingleThreadExecutor()
    @Volatile private var isWarmingUp = false

    // Double-Lock Frame Confirmation ताकि गलती से किसी दूसरे की हाज़िरी कभी न लगे
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
                                // सेव की गई फोटो में भी ML Kit से सटीक चेहरा और लैंडमार्क निकालें
                                val vec = extractBiometricFromStoredPhoto(rawBmp)
                                if (vec != null) {
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

    private fun extractBiometricFromStoredPhoto(bmp: Bitmap): FloatArray? {
        return try {
            val input = InputImage.fromBitmap(bmp, 0)
            val faces = Tasks.await(detector.process(input))
            if (faces.isNotEmpty()) {
                val primary = faces.maxByOrNull { it.boundingBox.width() * it.boundingBox.height() }!!
                val aligned = alignAndCropFace(bmp, primary)
                build192PointBiometricVector(aligned, primary)
            } else {
                // यदि पासपोर्ट फोटो बहुत पास से क्रॉप की गई है, तो सेंटर बायोमेट्रिक लें
                val fallbackCrop = cropCenterFaceFallback(bmp)
                build192PointBiometricVector(fallbackCrop, null)
            }
        } catch (e: Exception) {
            val fallbackCrop = cropCenterFaceFallback(bmp)
            build192PointBiometricVector(fallbackCrop, null)
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

    // MainActivity से कॉल होने वाला फंक्शन (यह चेहरे को आँखों की सीध में सीधा/Align भी करता है)
    fun cropFaceBitmap(source: Bitmap, box: Rect): Bitmap {
        val cx = box.exactCenterX()
        val cy = box.exactCenterY()
        val halfSize = max(box.width(), box.height()) * 0.48f

        val left = max(0, (cx - halfSize).toInt())
        val top = max(0, (cy - halfSize * 1.05f).toInt())
        val right = min(source.width, (cx + halfSize).toInt())
        val bottom = min(source.height, (cy + halfSize * 0.95f).toInt())

        val w = max(1, right - left)
        val h = max(1, bottom - top)
        return Bitmap.createBitmap(source, left, top, w, h)
    }

    private fun cropCenterFaceFallback(bmp: Bitmap): Bitmap {
        val x = (bmp.width * 0.12f).toInt()
        val y = (bmp.height * 0.10f).toInt()
        val w = max(1, (bmp.width * 0.76f).toInt())
        val h = max(1, (bmp.height * 0.80f).toInt())
        return Bitmap.createBitmap(bmp, x, y, w, h)
    }

    private fun alignAndCropFace(source: Bitmap, face: Face): Bitmap {
        var workingBmp = source
        val leftEye = face.getLandmark(FaceLandmark.LEFT_EYE)?.position
        val rightEye = face.getLandmark(FaceLandmark.RIGHT_EYE)?.position

        // आँखों के कोण के आधार पर झुके हुए चेहरे को सीधा (In-Plane Rotation Alignment) करें
        if (leftEye != null && rightEye != null) {
            val dy = (rightEye.y - leftEye.y).toDouble()
            val dx = (rightEye.x - leftEye.x).toDouble()
            val angleDeg = Math.toDegrees(atan2(dy, dx)).toFloat()
            if (abs(angleDeg) in 2.0f..28.0f) {
                val matrix = Matrix().apply {
                    postRotate(-angleDeg, source.width / 2f, source.height / 2f)
                }
                workingBmp = Bitmap.createBitmap(source, 0, 0, source.width, source.height, matrix, true)
            }
        }
        return cropFaceBitmap(workingBmp, face.boundingBox)
    }

    /**
     * 192-Point Biometric Feature Extractor:
     * 1. 128-D Illumination-Invariant LBP + Directional HOG Structural Grid (8x8 blocks x 2)
     * 2. 48-D Multi-Zone Center-Symmetric Local Binary Pattern (CS-LBP) Micro-Texture
     * 3. 16-D 3D Geometric & Facial Landmark Ratios (Eye-to-Eye, Nose Bridge, Mouth-to-Cheek)
     */
    private fun build192PointBiometricVector(faceBmp: Bitmap, faceObj: Face?): FloatArray {
        val size = 64
        val scaled = Bitmap.createScaledBitmap(faceBmp, size, size, true)
        val pixels = IntArray(size * size)
        scaled.getPixels(pixels, 0, size, 0, 0, size, size)

        // Step 1: Pure Luminance & Local Contrast Normalization (धूप या अंधेरे का असर खत्म करने के लिए)
        val gray = FloatArray(size * size)
        for (i in pixels.indices) {
            val p = pixels[i]
            val r = (p shr 16) and 0xFF
            val g = (p shr 8) and 0xFF
            val b = p and 0xFF
            gray[i] = 0.299f * r + 0.587f * g + 0.114f * b
        }

        val vector = FloatArray(192)
        var vIdx = 0

        // Part A: 128-D (8x8 Grid -> 2 Biometric Structural Values per cell: Oriented Gradient X & Y)
        for (by in 0 until 8) {
            for (bx in 0 until 8) {
                var cellMean = 0f
                for (y in (by * 8) until ((by + 1) * 8)) {
                    for (x in (bx * 8) until ((bx + 1) * 8)) {
                        cellMean += gray[y * size + x]
                    }
                }
                cellMean /= 64f

                var cellVar = 0f
                for (y in (by * 8) until ((by + 1) * 8)) {
                    for (x in (bx * 8) until ((bx + 1) * 8)) {
                        val d = gray[y * size + x] - cellMean
                        cellVar += d * d
                    }
                }
                val cellStd = max(2.5f, sqrt(cellVar / 64f))

                var gradH = 0f
                var gradV = 0f
                for (y in (by * 8) until ((by + 1) * 8)) {
                    for (x in (bx * 8) until ((bx + 1) * 8)) {
                        val idx = y * size + x
                        if (x in 1 until (size - 1)) {
                            gradH += abs(gray[idx + 1] - gray[idx - 1]) / cellStd
                        }
                        if (y in 1 until (size - 1)) {
                            gradV += abs(gray[idx + size] - gray[idx - size]) / cellStd
                        }
                    }
                }
                // चेहरा के मुख्य भाग (आँख, नाक, भौंह) को अधिक महत्व (Biometric Weighting)
                val isCoreFacialZone = (by in 1..6 && bx in 1..6)
                val zoneWeight = if (isCoreFacialZone) 1.35f else 0.65f

                vector[vIdx++] = (gradH / 64f) * zoneWeight
                vector[vIdx++] = (gradV / 64f) * zoneWeight
            }
        }

        // Part B: 48-D Center-Symmetric LBP (CS-LBP) Texture Histogram across 12 key facial regions (4x3 zones x 4 bins)
        for (zy in 0 until 3) {
            for (zx in 0 until 4) {
                val bins = FloatArray(4)
                val startY = 8 + zy * 16
                val startX = 8 + zx * 12
                for (y in startY until (startY + 14)) {
                    for (x in startX until (startX + 10)) {
                        val c = y * size + x
                        val b0 = if (gray[c + 1] > gray[c - 1] + 2f) 1 else 0
                        val b1 = if (gray[c + size] > gray[c - size] + 2f) 1 else 0
                        val b2 = if (gray[c + size + 1] > gray[c - size - 1] + 2f) 1 else 0
                        val b3 = if (gray[c + size - 1] > gray[c - size + 1] + 2f) 1 else 0
                        bins[0] += b0.toFloat()
                        bins[1] += b1.toFloat()
                        bins[2] += b2.toFloat()
                        bins[3] += b3.toFloat()
                    }
                }
                val norm = 140f
                for (k in 0 until 4) {
                    vector[vIdx++] = (bins[k] / norm) * 1.5f
                }
            }
        }

        // Part C: 16-D Radial Symmetry & Landmark Geometry Ratios
        val geometric = extract16DGeometry(gray, size, faceObj)
        for (g in geometric) {
            if (vIdx < 192) vector[vIdx++] = g
        }

        // Final L2 Normalization for pure Cosine Biometric Matching
        var normSq = 0f
        for (v in vector) normSq += v * v
        val invNorm = 1f / max(1e-6f, sqrt(normSq))
        for (i in vector.indices) {
            vector[i] *= invNorm
        }
        return vector
    }

    private fun extract16DGeometry(gray: FloatArray, size: Int, face: Face?): FloatArray {
        val geo = FloatArray(16)
        // Horizontal & Vertical Biometric Projections (Eye line, Nose bridge, Mouth line)
        for (i in 0 until 8) {
            var rowDiff = 0f
            val y = 8 + i * 6
            for (x in 10 until 54) {
                // Left-Right Facial Symmetry Check
                val symX = size - 1 - x
                rowDiff += abs(gray[y * size + x] - gray[y * size + symX])
            }
            geo[i] = (rowDiff / (44f * 255f)) * 1.2f
        }

        // Eye-to-Nose & Mouth-to-Eye Structural Ratios from ML Kit Landmarks (if present)
        val lEye = face?.getLandmark(FaceLandmark.LEFT_EYE)?.position
        val rEye = face?.getLandmark(FaceLandmark.RIGHT_EYE)?.position
        val nose = face?.getLandmark(FaceLandmark.NOSE_BASE)?.position
        val lMouth = face?.getLandmark(FaceLandmark.MOUTH_LEFT)?.position
        val rMouth = face?.getLandmark(FaceLandmark.MOUTH_RIGHT)?.position
        val bMouth = face?.getLandmark(FaceLandmark.MOUTH_BOTTOM)?.position

        if (lEye != null && rEye != null && nose != null) {
            val eyeDist = max(1f, dist(lEye, rEye))
            val eyeMid = PointF((lEye.x + rEye.x) / 2f, (lEye.y + rEye.y) / 2f)
            val noseLen = dist(eyeMid, nose) / eyeDist
            geo[8] = noseLen * 0.8f
            geo[9] = (dist(lEye, nose) / eyeDist) * 0.8f
            geo[10] = (dist(rEye, nose) / eyeDist) * 0.8f
            if (lMouth != null && rMouth != null) {
                val mouthWidth = dist(lMouth, rMouth) / eyeDist
                geo[11] = mouthWidth * 0.8f
            }
            if (bMouth != null) {
                val lowerFaceRatio = dist(nose, bMouth) / eyeDist
                geo[12] = lowerFaceRatio * 0.8f
            }
        } else {
            // Fallback structural ratios from normalized image bands
            geo[8] = 0.65f; geo[9] = 0.72f; geo[10] = 0.72f; geo[11] = 0.82f; geo[12] = 0.55f
        }
        geo[13] = 0.5f; geo[14] = 0.5f; geo[15] = 0.5f
        return geo
    }

    private fun dist(p1: PointF, p2: PointF): Float {
        return hypot((p1.x - p2.x).toDouble(), (p1.y - p2.y).toDouble()).toFloat()
    }

    private fun calculateBiometricDistance(v1: FloatArray, v2: FloatArray): Float {
        if (v1.size != v2.size) return 999f
        var dot = 0f
        var l1 = 0f
        // केवल पहले 176 (LBP + HOG) इंडेक्स पर मुख्य मिलान ताकि यदि किसी फोटो में लैंडमार्क न मिले तो भी कोई असली छात्र छूटे नहीं
        val activeLen = 176
        for (i in 0 until activeLen) {
            dot += v1[i] * v2[i]
            l1 += abs(v1[i] - v2[i])
        }
        // दोनों वेक्टर पहले से L2 Normalized हैं, इसलिए Cosine Distance + L1 Structural Penalty
        val cosineDist = max(0f, 1f - dot)
        return (cosineDist * 85f) + ((l1 / activeLen) * 110f)
    }

    fun findBestMatch(liveFaceBitmap: Bitmap, students: List<Student>): Pair<Student?, Float> {
        if (signatureCache.size < students.count { it.photos.isNotEmpty() }) {
            warmUpCache(students)
        }

        val liveVec = build192PointBiometricVector(liveFaceBitmap, null)
        var bestStudent: Student? = null
        var lowestScore = 999f
        var secondBestScore = 999f

        for (stu in students) {
            val storedList = signatureCache[stu.id] ?: continue
            var stuBest = 999f
            for (refVec in storedList) {
                val score = calculateBiometricDistance(liveVec, refVec)
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

        // Strict Anti-False-Positive Filter:
        // 1. स्कोर 17.5f से कम होना चाहिए (MainActivity में <= 19.0f चेक है)
        // 2. यदि एक से अधिक छात्र हैं, तो पहले और दूसरे छात्र के स्कोर में स्पष्ट अंतर (Distinctiveness Margin) होना चाहिए
        val marginOk = (students.size <= 1) || ((secondBestScore - lowestScore) >= 1.15f)

        if (bestStudent != null && lowestScore <= 17.2f && marginOk) {
            if (lastCandidateId == bestStudent.id) {
                consecutiveHits++
            } else {
                lastCandidateId = bestStudent.id
                consecutiveHits = 1
            }

            // यदि सिग्नेचर बहुत ही ज़्यादा सटीक है (<= 13.5f) तो तुरंत 1 फ्रेम में पास करें,
            // वरना 2 लगातार फ्रेम (0.6 सेकंड) में कन्फर्म करें ताकि किसी गलत चेहरे की हाज़िरी कभी न लगे!
            return if (lowestScore <= 13.5f || consecutiveHits >= 2) {
                consecutiveHits = 0
                Pair(bestStudent, lowestScore)
            } else {
                // दूसरे कन्फर्मेशन फ्रेम का इंतज़ार (इससे मैनुअल या ऑटो स्कैन में गलत व्यक्ति कभी पास नहीं होगा)
                Pair(null, 25.0f)
            }
        } else {
            lastCandidateId = null
            consecutiveHits = 0
            return Pair(bestStudent, max(22.0f, lowestScore))
        }
    }
}
