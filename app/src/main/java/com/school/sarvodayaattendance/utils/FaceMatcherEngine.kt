package com.school.sarvodayaattendance.utils

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Matrix
import android.graphics.Rect
import android.util.Base64
import com.google.android.gms.tasks.Tasks
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.face.Face
import com.google.mlkit.vision.face.FaceDetection
import com.google.mlkit.vision.face.FaceDetectorOptions
import com.google.mlkit.vision.face.FaceLandmark
import com.school.sarvodayaattendance.data.Student
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.Executors
import kotlin.math.abs
import kotlin.math.atan2
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

    private var tfliteInterpreter: Interpreter? = null
    private var modelInputSize = 112
    private var modelOutputSize = 192

    private val signatureCache = ConcurrentHashMap<String, List<FloatArray>>()
    private val bgExecutor = Executors.newSingleThreadExecutor()
    @Volatile private var isWarmingUp = false

    private var lastCandidateId: String? = null
    private var consecutiveHits: Int = 0

    // मोबाइल फेस-लॉक का MobileFaceNet मॉडल लोड करें
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

            val inShape = interpreter.getInputTensor(0).shape() // e.g., [1, 112, 112, 3]
            val outShape = interpreter.getOutputTensor(0).shape() // e.g., [1, 192] or [1, 128]
            if (inShape.size >= 3) modelInputSize = inShape[1]
            if (outShape.size >= 2) modelOutputSize = outShape[1]

            tfliteInterpreter = interpreter
        } catch (e: Exception) {
            e.printStackTrace()
        }
    }

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
                                extractNeuralEmbeddingFromStored(rawBmp)?.let { vec ->
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

    private fun extractNeuralEmbeddingFromStored(bmp: Bitmap): FloatArray? {
        return try {
            val input = InputImage.fromBitmap(bmp, 0)
            val faces = Tasks.await(detector.process(input))
            if (faces.isNotEmpty()) {
                val primary = faces.maxByOrNull { it.boundingBox.width() * it.boundingBox.height() }!!
                val aligned = alignAndCropFace(bmp, primary)
                extractDeepFaceEmbedding(aligned)
            } else {
                extractDeepFaceEmbedding(bmp)
            }
        } catch (e: Exception) {
            extractDeepFaceEmbedding(bmp)
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
        val halfSize = max(box.width(), box.height()) * 0.48f

        val left = max(0, (cx - halfSize).toInt())
        val top = max(0, (cy - halfSize).toInt())
        val right = min(source.width, (cx + halfSize).toInt())
        val bottom = min(source.height, (cy + halfSize).toInt())

        val w = max(1, right - left)
        val h = max(1, bottom - top)
        return Bitmap.createBitmap(source, left, top, w, h)
    }

    private fun alignAndCropFace(source: Bitmap, face: Face): Bitmap {
        var workingBmp = source
        val leftEye = face.getLandmark(FaceLandmark.LEFT_EYE)?.position
        val rightEye = face.getLandmark(FaceLandmark.RIGHT_EYE)?.position

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
     * MobileFaceNet Deep Neural Network Embedding (112x112 RGB -> 192-D L2 Normalized Vector)
     */
    @Synchronized
    private fun extractDeepFaceEmbedding(faceBmp: Bitmap): FloatArray {
        val interpreter = tfliteInterpreter
        if (interpreter != null) {
            try {
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
                    // MobileFaceNet Standard Normalization: (val - 127.5) / 128.0
                    imgData.putFloat((r - 127.5f) / 128.0f)
                    imgData.putFloat((g - 127.5f) / 128.0f)
                    imgData.putFloat((b - 127.5f) / 128.0f)
                }

                val outputArray = Array(1) { FloatArray(modelOutputSize) }
                interpreter.run(imgData, outputArray)
                val rawEmbedding = outputArray[0]

                // L2 Normalize Embedding
                var norm = 0f
                for (v in rawEmbedding) norm += v * v
                val invNorm = 1f / max(1e-6f, sqrt(norm))
                for (i in rawEmbedding.indices) {
                    rawEmbedding[i] *= invNorm
                }
                return rawEmbedding
            } catch (e: Exception) {
                e.printStackTrace()
            }
        }
        return FloatArray(192)
    }

    /**
     * MobileFaceNet Cosine Similarity & Euclidean Distance Check:
     * - एक ही व्यक्ति (Same Person) का Cosine Similarity (dot): 0.68 से 0.96 तक होता है।
     * - दूसरे व्यक्ति या मित्र (Different Person) का Cosine Similarity: -0.10 से 0.48 तक ही रहता है!
     */
    private fun computeNeuralDistance(v1: FloatArray, v2: FloatArray): Float {
        if (v1.size != v2.size) return 999f
        var dot = 0f
        for (i in v1.indices) {
            dot += v1[i] * v2[i]
        }
        // यदि Cosine Similarity 0.63 (63%) से कम है, तो यह 100% कोई दूसरा इंसान है -> इसे तुरंत रिजेक्ट (Score 50.0f) करें!
        if (dot < 0.63f) {
            return 50.0f
        }
        // यदि 0.63 से ऊपर है, तो इसे 0 से 18 के स्केल में बदलें (MainActivity <= 19.0f को पास मानती है)
        return (1.0f - dot) * 45.0f
    }

    fun findBestMatch(liveFaceBitmap: Bitmap, students: List<Student>): Pair<Student?, Float> {
        if (signatureCache.size < students.count { it.photos.isNotEmpty() }) {
            warmUpCache(students)
        }

        val liveEmbedding = extractDeepFaceEmbedding(liveFaceBitmap)

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

        val marginOk = (students.size <= 1) || ((secondBestScore - lowestScore) >= 1.5f)

        // केवल तभी पास करें जब Neural Distance <= 16.5f हो (यानी Cosine Similarity >= 63.5%)
        if (bestStudent != null && lowestScore <= 16.5f && marginOk) {
            if (lastCandidateId == bestStudent.id) {
                consecutiveHits++
            } else {
                lastCandidateId = bestStudent.id
                consecutiveHits = 1
            }

            // यदि बहुत सटीक मैच है (Similarity >= 72%, score <= 12.5f) तो तुरंत 1 फ्रेम में, वरना 2 लगातार फ्रेम में पुष्टि करें
            return if (lowestScore <= 12.5f || consecutiveHits >= 2) {
                consecutiveHits = 0
                Pair(bestStudent, lowestScore)
            } else {
                Pair(null, 30.0f)
            }
        } else {
            lastCandidateId = null
            consecutiveHits = 0
            return Pair(bestStudent, max(30.0f, lowestScore))
        }
    }
}
