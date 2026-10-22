package com.school.sarvodayaattendance.utils

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Rect
import android.util.Base64
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.face.FaceDetection
import com.google.mlkit.vision.face.FaceDetectorOptions
import com.school.sarvodayaattendance.data.Student
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

object FaceMatcherEngine {

    private val detectorOptions = FaceDetectorOptions.Builder()
        .setPerformanceMode(FaceDetectorOptions.PERFORMANCE_MODE_FAST)
        .setLandmarkMode(FaceDetectorOptions.LANDMARK_MODE_ALL)
        .setClassificationMode(FaceDetectorOptions.CLASSIFICATION_MODE_ALL)
        .setMinFaceSize(0.18f)
        .build()

    val detector = FaceDetection.getClient(detectorOptions)

    // RAM Cache for 0.1ms instant matching of 100+ students
    private val signatureCache = mutableMapOf<String, List<FloatArray>>()

    fun clearCache() {
        signatureCache.clear()
    }

    fun warmUpCache(students: List<Student>) {
        for (stu in students) {
            if (!signatureCache.containsKey(stu.id) && stu.photos.isNotEmpty()) {
                val vectors = mutableListOf<FloatArray>()
                for (b64 in stu.photos) {
                    decodeBase64Bitmap(b64)?.let { bmp ->
                        vectors.add(extractFaceSignature(bmp, true))
                    }
                }
                if (vectors.isNotEmpty()) {
                    signatureCache[stu.id] = vectors
                }
            }
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

    fun cropFaceBitmap(source: Bitmap, box: Rect): Bitmap {
        val padX = (box.width() * 0.08f).toInt()
        val padY = (box.height() * 0.08f).toInt()
        val left = max(0, box.left - padX)
        val top = max(0, box.top - padY)
        val right = min(source.width, box.right + padX)
        val bottom = min(source.height, box.bottom + padY)
        val w = max(1, right - left)
        val h = max(1, bottom - top)
        return Bitmap.createBitmap(source, left, top, w, h)
    }

    fun extractFaceSignature(bmp: Bitmap, isStoredPhoto: Boolean): FloatArray {
        val cropped = if (isStoredPhoto) {
            val x = (bmp.width * 0.15f).toInt()
            val y = (bmp.height * 0.12f).toInt()
            val w = max(1, (bmp.width * 0.70f).toInt())
            val h = max(1, (bmp.height * 0.76f).toInt())
            Bitmap.createBitmap(bmp, x, y, w, h)
        } else {
            bmp
        }

        val scaled = Bitmap.createScaledBitmap(cropped, 48, 48, true)
        val pixels = IntArray(48 * 48)
        scaled.getPixels(pixels, 0, 48, 0, 0, 48, 48)

        val gray = FloatArray(48 * 48)
        var mean = 0f
        for (i in pixels.indices) {
            val p = pixels[i]
            val r = (p shr 16) and 0xFF
            val g = (p shr 8) and 0xFF
            val b = p and 0xFF
            val lum = 0.299f * r + 0.587f * g + 0.114f * b
            gray[i] = lum
            mean += lum
        }
        mean /= gray.size

        var variance = 0f
        for (v in gray) {
            val d = v - mean
            variance += d * d
        }
        val std = max(1f, sqrt(variance / gray.size))

        val vector = FloatArray(8 * 8 * 5)
        var vIdx = 0

        for (by in 0 until 8) {
            for (bx in 0 until 8) {
                var blockLum = 0f
                var gxSum = 0f
                var gySum = 0f
                var rSum = 0f
                var gSum = 0f
                var bSum = 0f

                for (y in (by * 6) until ((by + 1) * 6)) {
                    for (x in (bx * 6) until ((bx + 1) * 6)) {
                        val idx = y * 48 + x
                        blockLum += (gray[idx] - mean) / std
                        if (x < 47) gxSum += abs(gray[idx + 1] - gray[idx])
                        if (y < 47) gySum += abs(gray[idx + 48] - gray[idx])

                        val px = pixels[idx]
                        rSum += (px shr 16) and 0xFF
                        gSum += (px shr 8) and 0xFF
                        bSum += px and 0xFF
                    }
                }
                val rgbTotal = max(1f, rSum + gSum + bSum)
                vector[vIdx++] = blockLum / 36f
                vector[vIdx++] = gxSum / (36f * std)
                vector[vIdx++] = gySum / (36f * std)
                vector[vIdx++] = (rSum / rgbTotal) * 3f
                vector[vIdx++] = (gSum / rgbTotal) * 3f
            }
        }
        return vector
    }

    private fun compareVectors(v1: FloatArray, v2: FloatArray): Float {
        if (v1.size != v2.size) return 999f
        var l1 = 0f
        var dot = 0f
        var n1 = 0f
        var n2 = 0f
        for (i in v1.indices) {
            l1 += abs(v1[i] - v2[i])
            dot += v1[i] * v2[i]
            n1 += v1[i] * v1[i]
            n2 += v2[i] * v2[i]
        }
        val denom = max(1e-6f, sqrt(n1) * sqrt(n2))
        val cosine = dot / denom
        return (1f - cosine) * 100f + (l1 / v1.size) * 25f
    }

    fun findBestMatch(liveFaceBitmap: Bitmap, students: List<Student>): Pair<Student?, Float> {
        warmUpCache(students)
        val liveVec = extractFaceSignature(liveFaceBitmap, false)
        var bestStudent: Student? = null
        var lowestScore = 999f

        for (stu in students) {
            val storedList = signatureCache[stu.id] ?: continue
            for (refVec in storedList) {
                val score = compareVectors(liveVec, refVec)
                if (score < lowestScore) {
                    lowestScore = score
                    bestStudent = stu
                }
            }
        }
        return Pair(bestStudent, lowestScore)
    }
}
