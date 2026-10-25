package com.school.sarvodayaattendance.utils

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Rect
import android.util.Base64
import com.google.mlkit.vision.face.FaceDetection
import com.google.mlkit.vision.face.FaceDetectorOptions
import com.school.sarvodayaattendance.data.Student
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import java.util.concurrent.ConcurrentHashMap
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

object FaceMatcherEngine {

    private val detectorOptions = FaceDetectorOptions.Builder()
        .setPerformanceMode(FaceDetectorOptions.PERFORMANCE_MODE_FAST)
        .setLandmarkMode(FaceDetectorOptions.LANDMARK_MODE_ALL)
        .setClassificationMode(FaceDetectorOptions.CLASSIFICATION_MODE_NONE)
        .setMinFaceSize(0.18f)
        .build()

    val detector = FaceDetection.getClient(detectorOptions)

    private var tfliteInterpreter: Interpreter? = null
    private var modelInputSize = 112
    private var modelOutputSize = 192

    // RAM Cache: Student ID -> List of 192-Point Neural Embeddings
    private val signatureCache = ConcurrentHashMap<String, List<FloatArray>>()

    @Synchronized
    fun initEngine(context: Context) {
        if (tfliteInterpreter != null) return
        try {
            val afd = context.assets.openFd("mobile_face_net.tflite")
            val inputStream = FileInputStream(afd.fileDescriptor)
            val fileChannel = inputStream.channel
            val mappedByteBuffer: MappedByteBuffer = fileChannel.map(
                FileChannel.MapMode.READ_ONLY,
                afd.startOffset,
                afd.declaredLength
            )
            val options = Interpreter.Options().apply { setNumThreads(4) }
            val interpreter = Interpreter(mappedByteBuffer, options)

            val inShape = interpreter.getInputTensor(0).shape()   // [1, 112, 112, 3]
            val outShape = interpreter.getOutputTensor(0).shape() // [1, 192]
            if (inShape.size >= 3) modelInputSize = inShape[1]
            if (outShape.size >= 2) modelOutputSize = outShape[1]

            tfliteInterpreter = interpreter
        } catch (e: Exception) {
            e.printStackTrace()
        }
    }

    fun clearCache() {
        signatureCache.clear()
    }

    fun warmUpCache(students: List<Student>) {
        if (tfliteInterpreter == null) return
        try {
            for (stu in students) {
                if (!signatureCache.containsKey(stu.id) && stu.photos.isNotEmpty()) {
                    val vectors = mutableListOf<FloatArray>()
                    for (b64 in stu.photos) {
                        decodeBase64Bitmap(b64)?.let { bmp ->
                            val vec = extractPureNeuralEmbedding(bmp)
                            if (vec.isNotEmpty()) {
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
        val cx = box.exactCenterX()
        val cy = box.exactCenterY()
        val halfSize = max(box.width(), box.height()) * 0.46f

        val left = max(0, (cx - halfSize).toInt())
        val top = max(0, (cy - halfSize).toInt())
        val right = min(source.width, (cx + halfSize).toInt())
        val bottom = min(source.height, (cy + halfSize).toInt())

        val w = max(1, right - left)
        val h = max(1, bottom - top)
        return Bitmap.createBitmap(source, left, top, w, h)
    }

    /**
     * 100% Pure MobileFaceNet 192-Point Deep Neural Embedding Extractor (No 2D Fallback)
     */
    @Synchronized
    private fun extractPureNeuralEmbedding(faceBmp: Bitmap): FloatArray {
        val interpreter = tfliteInterpreter ?: return FloatArray(0)
        return try {
            val scaled = Bitmap.createScaledBitmap(faceBmp, modelInputSize, modelInputSize, true)
            val imgData = ByteBuffer.allocateDirect(1 * modelInputSize * modelInputSize * 3 * 4)
            imgData.order(ByteOrder.nativeOrder())
            imgData.rewind()

            val intValues = IntArray(modelInputSize * modelInputSize)
            scaled.getPixels(intValues, 0, modelInputSize, 0, 0, modelInputSize, modelInputSize)

            for (pixelValue in intValues) {
                val r = ((pixelValue shr 16) and 0xFF)
                val g = ((pixelValue shr 8) and 0xFF)
                val b = (pixelValue and 0xFF)
                // Standard MobileFaceNet Tensor Normalization: (val - 127.5f) / 128.0f
                imgData.putFloat((r - 127.5f) / 128.0f)
                imgData.putFloat((g - 127.5f) / 128.0f)
                imgData.putFloat((b - 127.5f) / 128.0f)
            }

            val outputArray = Array(1) { FloatArray(modelOutputSize) }
            interpreter.run(imgData, outputArray)
            val rawEmbedding = outputArray[0]

            // L2 Normalization for exact Cosine Similarity & Euclidean Distance
            var normSq = 0f
            for (v in rawEmbedding) normSq += v * v
            if (normSq < 1e-6f) return FloatArray(0)

            val invNorm = 1f / sqrt(normSq)
            for (i in rawEmbedding.indices) {
                rawEmbedding[i] *= invNorm
            }
            rawEmbedding
        } catch (e: Exception) {
            e.printStackTrace()
            FloatArray(0)
        }
    }

    /**
     * Pure 192-Point Neural Cosine Similarity Comparison:
     * - असली छात्र (Same Person): Cosine Similarity (dot) >= 0.58 होता है -> Score <= 16.8f (Pass)
     * - मित्र या दूसरा व्यक्ति (Different Person): Cosine Similarity < 0.55 होता है -> Score = 50.0f (Strict Reject)
     */
    private fun computeNeuralDistance(v1: FloatArray, v2: FloatArray): Float {
        if (v1.isEmpty() || v2.isEmpty() || v1.size != v2.size) return 999f
        var dot = 0f
        for (i in v1.indices) {
            dot += v1[i] * v2[i]
        }
        // यदि 192-पॉइंट न्यूरल समानता 0.58 (58%) से कम है, तो यह दूसरा व्यक्ति है — तुरंत रिजेक्ट करें
        if (dot < 0.58f) {
            return 50.0f
        }
        return (1.0f - dot) * 40.0f
    }

    fun findBestMatch(liveFaceBitmap: Bitmap, students: List<Student>): Pair<Student?, Float> {
        if (tfliteInterpreter == null) return Pair(null, 999f)
        warmUpCache(students)

        val liveEmbedding = extractPureNeuralEmbedding(liveFaceBitmap)
        if (liveEmbedding.isEmpty()) return Pair(null, 999f)

        var bestStudent: Student? = null
        var lowestScore = 999f
        var secondBestScore = 999f

        for (stu in students) {
            val storedList = signatureCache[stu.id] ?: continue
            var stuBest = 999f
            for (refVec in storedList) {
                val score = computeNeuralDistance(liveEmbedding, refVec)
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

        val marginOk = (students.size <= 1) || ((secondBestScore - lowestScore) >= 1.4f)

        return if (bestStudent != null && lowestScore <= 16.8f && marginOk) {
            Pair(bestStudent, lowestScore)
        } else {
            Pair(bestStudent, max(30.0f, lowestScore))
        }
    }
}
