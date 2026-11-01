package com.school.sarvodayaattendance

import android.Manifest
import android.annotation.SuppressLint
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Color
import android.graphics.Matrix
import android.graphics.Typeface
import android.graphics.drawable.GradientDrawable
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.text.InputType
import android.util.Size
import android.view.Gravity
import android.view.View
import android.view.ViewGroup
import android.view.WindowManager
import android.widget.Button
import android.widget.EditText
import android.widget.FrameLayout
import android.widget.LinearLayout
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.Camera
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import com.google.mlkit.vision.common.InputImage
import com.school.sarvodayaattendance.data.AppStorage
import com.school.sarvodayaattendance.data.BackupPayload
import com.school.sarvodayaattendance.data.Student
import com.school.sarvodayaattendance.utils.FaceMatcherEngine
import com.school.sarvodayaattendance.utils.RealisticVoiceManager
import java.util.concurrent.Executors
import kotlin.math.abs

class MainActivity : AppCompatActivity() {

    private lateinit var previewView: PreviewView
    private lateinit var statusBanner: TextView
    private lateinit var statsChip: TextView
    private lateinit var classFilterBtn: Button
    private lateinit var torchBtn: Button
    private lateinit var voiceBtn: Button

    private lateinit var voiceManager: RealisticVoiceManager
    private val cameraExecutor = Executors.newSingleThreadExecutor()
    private val aiVerifyExecutor = Executors.newSingleThreadExecutor()
    private val mainHandler = Handler(Looper.getMainLooper())

    private var lensFacing = CameraSelector.LENS_FACING_FRONT
    private var activeCamera: Camera? = null
    private var isTorchOn = false

    private val classOptions = listOf("सभी", "6वीं", "7वीं", "8वीं")
    private var selectedClassFilter = "सभी"

    private var currentPayload: BackupPayload = BackupPayload()

    @Volatile
    private var isProcessingFrame = false
    private var lastFrameCheckMs = 0L

    private var lastRecognizedId: String? = null
    private var lastRecognizedTimeMs = 0L
    private var samePersonStreak = 0
    private var lastUnknownVoiceTimeMs = 0L

    // ✅ लगातार 2 फ्रेम की पुष्टि (2-Frame Consecutive Confirmation) के लिए वेरिएबल्स
    private var pendingCandidateId: String? = null
    private var pendingCandidateHits: Int = 0
    private var pendingCandidateLastSeenMs: Long = 0L

    @Volatile
    private var isOnlineAiVerifying = false

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)

        FaceMatcherEngine.initEngine(this)
        voiceManager = RealisticVoiceManager(this)

        buildScannerScreenUI()
        reloadPayloadAndRefreshUI()

        mainHandler.postDelayed({
            voiceManager.speakStartupWelcome()
        }, 700)

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
            startCamera()
        } else {
            ActivityCompat.requestPermissions(this, arrayOf(Manifest.permission.CAMERA), 101)
        }
    }

    override fun onResume() {
        super.onResume()
        reloadPayloadAndRefreshUI()
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraExecutor.shutdown()
        aiVerifyExecutor.shutdown()
        voiceManager.shutdown()
    }

    private fun reloadPayloadAndRefreshUI() {
        currentPayload = AppStorage.loadData(this)
        if (currentPayload.sharedApiKeys.isNotBlank() && voiceManager.getApiKey().isBlank()) {
            voiceManager.saveCustomApiKey(currentPayload.sharedApiKeys)
        }
        FaceMatcherEngine.warmUpCache(currentPayload.students)
        updateStatsBar()
    }

    private fun getFilteredCandidateStudents(): List<Student> {
        return if (selectedClassFilter == "सभी") {
            currentPayload.students
        } else {
            currentPayload.students.filter { it.className == selectedClassFilter }
        }
    }

    // ✅ सटीक उपस्थिति गिनती: केवल वही आईडी गिनी जाएंगी जो वास्तव में वर्तमान छात्र सूची में मौजूद हैं
    private fun getAccuratePresentAndTotalCount(studentsList: List<Student>): Pair<Int, Int> {
        val todayKey = AppStorage.getTodayShortKey()
        val presentSet = (currentPayload.attendance[todayKey] ?: emptyList()).toSet()
        val validPresentCount = studentsList.count { presentSet.contains(it.id) }
        return Pair(validPresentCount, studentsList.size)
    }

    private fun updateStatsBar() {
        val list = getFilteredCandidateStudents()
        val (presentCount, totalCount) = getAccuratePresentAndTotalCount(list)
        val pStr = presentCount.toString().padStart(2, '0')
        val tStr = totalCount.toString().padStart(2, '0')
        statsChip.text = "📊 $selectedClassFilter: $pStr / $tStr उपस्थित"
    }

    private fun buildScannerScreenUI() {
        val root = FrameLayout(this).apply {
            setBackgroundColor(Color.parseColor("#0F172A"))
        }

        previewView = PreviewView(this).apply {
            implementationMode = PreviewView.ImplementationMode.COMPATIBLE
            scaleType = PreviewView.ScaleType.FILL_CENTER
        }
        root.addView(
            previewView,
            FrameLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.MATCH_PARENT
            )
        )

        val topOverlay = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(28, 42, 28, 22)
            background = GradientDrawable(
                GradientDrawable.Orientation.TOP_BOTTOM,
                intArrayOf(0xDD0F172A.toInt(), 0x000F172A)
            )
        }

        val headerRow = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
        }

        val titleView = TextView(this).apply {
            text = "सर्वोदय मध्य विद्यालय, सोहसराय"
            setTextColor(Color.WHITE)
            textSize = 16.5f
            setTypeface(typeface, Typeface.BOLD)
            layoutParams = LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f)
            setOnLongClickListener {
                val (presentCount, totalCount) = getAccuratePresentAndTotalCount(currentPayload.students)
                voiceManager.speakVipInspectionWelcome(presentCount, totalCount)
                Toast.makeText(this@MainActivity, "🎙️ VIP निरीक्षण स्वागत...", Toast.LENGTH_SHORT).show()
                true
            }
        }

        val adminBtn = Button(this).apply {
            text = "⚙️ एडमिन"
            setTextColor(Color.WHITE)
            textSize = 13.5f
            setTypeface(typeface, Typeface.BOLD)
            setPadding(30, 12, 30, 12)
            background = GradientDrawable().apply {
                setColor(Color.parseColor("#1E3A8A"))
                cornerRadius = 999f
                setStroke(2, Color.parseColor("#93C5FD"))
            }
            setOnClickListener {
                promptAdminPinAndOpen()
            }
        }

        headerRow.addView(titleView)
        headerRow.addView(adminBtn)

        val subRow = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
            setPadding(0, 18, 0, 0)
        }

        statsChip = TextView(this).apply {
            text = "📊 सभी: 00 / 00 उपस्थित"
            setTextColor(Color.parseColor("#FDE047"))
            textSize = 14f
            setTypeface(typeface, Typeface.BOLD)
            layoutParams = LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f)
        }

        classFilterBtn = Button(this).apply {
            text = "कक्षा: सभी ▾"
            setTextColor(Color.WHITE)
            textSize = 13f
            setTypeface(typeface, Typeface.BOLD)
            setPadding(26, 10, 26, 10)
            background = GradientDrawable().apply {
                setColor(0x55FFFFFF)
                cornerRadius = 999f
            }
            setOnClickListener {
                val nextIdx = (classOptions.indexOf(selectedClassFilter) + 1) % classOptions.size
                selectedClassFilter = classOptions[nextIdx]
                text = "कक्षा: $selectedClassFilter ▾"
                pendingCandidateId = null
                pendingCandidateHits = 0
                updateStatsBar()
            }
        }

        subRow.addView(statsChip)
        subRow.addView(classFilterBtn)

        topOverlay.addView(headerRow)
        topOverlay.addView(subRow)

        root.addView(
            topOverlay,
            FrameLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT,
                Gravity.TOP
            )
        )

        val bottomOverlay = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            gravity = Gravity.CENTER_HORIZONTAL
            setPadding(32, 24, 32, 46)
            background = GradientDrawable(
                GradientDrawable.Orientation.BOTTOM_TOP,
                intArrayOf(0xEE0F172A.toInt(), 0x000F172A)
            )
        }

        statusBanner = TextView(this).apply {
            text = "📷 कैमरे के सामने सीधे देखें..."
            setTextColor(Color.WHITE)
            textSize = 16.5f
            gravity = Gravity.CENTER
            setTypeface(typeface, Typeface.BOLD)
            setPadding(36, 26, 36, 26)
            background = GradientDrawable().apply {
                setColor(Color.parseColor("#1E293B"))
                cornerRadius = 32f
                setStroke(3, Color.parseColor("#38BDF8"))
            }
        }

        val controlsRow = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER
            setPadding(0, 26, 0, 0)
        }

        val flipCamBtn = createControlPillButton("🔄 कैमरा पलटें") {
            lensFacing = if (lensFacing == CameraSelector.LENS_FACING_FRONT) {
                CameraSelector.LENS_FACING_BACK
            } else {
                CameraSelector.LENS_FACING_FRONT
            }
            isTorchOn = false
            startCamera()
        }

        torchBtn = createControlPillButton("🔦 फ्लैश: बंद") {
            val cam = activeCamera ?: return@createControlPillButton
            if (cam.cameraInfo.hasFlashUnit()) {
                isTorchOn = !isTorchOn
                cam.cameraControl.enableTorch(isTorchOn)
                torchBtn.text = if (isTorchOn) "🔦 फ्लैश: चालू" else "🔦 फ्लैश: बंद"
            } else {
                Toast.makeText(this, "इस कैमरे में फ्लैश नहीं है", Toast.LENGTH_SHORT).show()
            }
        }

        voiceBtn = createControlPillButton("🔊 आवाज़: चालू") {
            voiceManager.isVoiceEnabled = !voiceManager.isVoiceEnabled
            voiceBtn.text = if (voiceManager.isVoiceEnabled) "🔊 आवाज़: चालू" else "🔇 आवाज़: बंद"
        }

        controlsRow.addView(flipCamBtn)
        controlsRow.addView(View(this).apply { layoutParams = LinearLayout.LayoutParams(18, 1) })
        controlsRow.addView(torchBtn)
        controlsRow.addView(View(this).apply { layoutParams = LinearLayout.LayoutParams(18, 1) })
        controlsRow.addView(voiceBtn)

        bottomOverlay.addView(
            statusBanner,
            LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT
            )
        )
        bottomOverlay.addView(controlsRow)

        root.addView(
            bottomOverlay,
            FrameLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT,
                Gravity.BOTTOM
            )
        )

        setContentView(root)
    }

    private fun createControlPillButton(label: String, onClick: () -> Unit): Button {
        return Button(this).apply {
            text = label
            setTextColor(Color.WHITE)
            textSize = 12.5f
            setTypeface(typeface, Typeface.BOLD)
            setPadding(28, 16, 28, 16)
            background = GradientDrawable().apply {
                setColor(Color.parseColor("#334155"))
                cornerRadius = 999f
                setStroke(2, Color.parseColor("#64748B"))
            }
            setOnClickListener { onClick() }
        }
    }

    private fun promptAdminPinAndOpen() {
        val input = EditText(this).apply {
            inputType = InputType.TYPE_CLASS_NUMBER or InputType.TYPE_NUMBER_VARIATION_PASSWORD
            hint = "6 अंकों का पिन डालें"
            gravity = Gravity.CENTER
            textSize = 18f
        }

        AlertDialog.Builder(this)
            .setTitle("🔒 एडमिन लॉगिन")
            .setView(input)
            .setPositiveButton("खोलें") { _, _ ->
                val entered = input.text.toString().trim()
                val savedPin = currentPayload.adminPin.ifBlank { "123456" }
                if (entered == savedPin) {
                    startActivity(Intent(this, AdminActivity::class.java))
                } else {
                    Toast.makeText(this, "❌ गलत पिन!", Toast.LENGTH_SHORT).show()
                }
            }
            .setNegativeButton("रद्द करें", null)
            .show()
    }

    private fun startCamera() {
        val providerFuture = ProcessCameraProvider.getInstance(this)
        providerFuture.addListener({
            try {
                val provider = providerFuture.get()
                val preview = Preview.Builder().build().also {
                    it.setSurfaceProvider(previewView.surfaceProvider)
                }

                val analyzer = ImageAnalysis.Builder()
                    .setTargetResolution(Size(640, 480))
                    .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                    .build()
                    .also { analysis ->
                        analysis.setAnalyzer(cameraExecutor) { imageProxy ->
                            processCameraFrame(imageProxy)
                        }
                    }

                val selector = CameraSelector.Builder().requireLensFacing(lensFacing).build()
                provider.unbindAll()
                activeCamera = provider.bindToLifecycle(this, selector, preview, analyzer)
            } catch (e: Exception) {
                e.printStackTrace()
            }
        }, ContextCompat.getMainExecutor(this))
    }

    @SuppressLint("UnsafeOptInUsageError")
    private fun processCameraFrame(imageProxy: ImageProxy) {
        val now = System.currentTimeMillis()
        if (isProcessingFrame || isOnlineAiVerifying || now - lastFrameCheckMs < 300) {
            imageProxy.close()
            return
        }

        val mediaImage = imageProxy.image
        if (mediaImage == null) {
            imageProxy.close()
            return
        }

        isProcessingFrame = true
        lastFrameCheckMs = now

        val rotation = imageProxy.imageInfo.rotationDegrees
        val inputImage = InputImage.fromMediaImage(mediaImage, rotation)

        FaceMatcherEngine.detector.process(inputImage)
            .addOnSuccessListener { faces ->
                if (faces.isEmpty()) {
                    pendingCandidateId = null
                    pendingCandidateHits = 0
                    isProcessingFrame = false
                    imageProxy.close()
                    return@addOnSuccessListener
                }

                val primaryFace = faces.maxByOrNull { it.boundingBox.width() * it.boundingBox.height() }!!

                // चेहरा बहुत ज़्यादा तिरछा हो तो गलत पहचान रोकने के लिए सीधा देखने को कहें
                if (abs(primaryFace.headEulerAngleY) > 22f || abs(primaryFace.headEulerAngleZ) > 20f) {
                    mainHandler.post {
                        updateBanner("👀 कृपया कैमरे की तरफ सीधा देखें...", "#334155", "#94A3B8")
                    }
                    isProcessingFrame = false
                    imageProxy.close()
                    return@addOnSuccessListener
                }

                mainHandler.post {
                    val frameBmp = previewView.bitmap
                    if (frameBmp != null && primaryFace.boundingBox.width() >= 65) {
                        val alignedFaceBmp = FaceMatcherEngine.alignAndCropFaceBitmap(frameBmp, primaryFace)
                        val candidates = getFilteredCandidateStudents()

                        cameraExecutor.execute {
                            val matchResult = FaceMatcherEngine.findBestMatchDetailed(alignedFaceBmp, candidates)
                            handleStrictBiometricMatchResult(matchResult, alignedFaceBmp)
                            isProcessingFrame = false
                        }
                    } else {
                        isProcessingFrame = false
                    }
                }
                imageProxy.close()
            }
            .addOnFailureListener {
                isProcessingFrame = false
                imageProxy.close()
            }
    }

    // ✅ 2-Frame Consecutive Confirmation + Online Gemini Vision AI Gender/Identity Double-Check
    private fun handleStrictBiometricMatchResult(
        result: FaceMatcherEngine.MatchResult,
        liveFaceBmp: Bitmap
    ) {
        val now = System.currentTimeMillis()
        val matchedStudent = result.student

        if (matchedStudent != null) {
            if (pendingCandidateId == matchedStudent.id && (now - pendingCandidateLastSeenMs) < 2200L) {
                pendingCandidateHits++
            } else {
                pendingCandidateId = matchedStudent.id
                pendingCandidateHits = 1
            }
            pendingCandidateLastSeenMs = now

            // पहले फ्रेम पर केवल "जाँच हो रही है..." दिखाएं, तुक्के से तुरंत नाम लॉक न करें
            if (pendingCandidateHits < 2) {
                mainHandler.post {
                    updateBanner("🔍 चेहरे की पुष्टि हो रही है... स्थिर रहें", "#1E3A8A", "#60A5FA")
                }
                return
            }

            // यदि 2 लगातार फ्रेम में वही छात्र आया है:
            // यदि यह बॉर्डरलाइन केस है या बहुत हाई-कॉन्फिडेंस नहीं है, तो Online Gemini Vision AI से लड़का/लड़की और चेहरे की 1-to-1 पुष्टि करें
            if (result.isBorderlineForAiCheck || result.distance > 8.8f) {
                val hasGeminiKey = voiceManager.getGeminiKeysString().trim().length > 15
                if (hasGeminiKey && !isOnlineAiVerifying) {
                    isOnlineAiVerifying = true
                    mainHandler.post {
                        updateBanner("🤖 AI से चेहरे की सटीक जाँच हो रही है...", "#1E3A8A", "#38BDF8")
                    }
                    val frozenBmp = liveFaceBmp.copy(Bitmap.Config.ARGB_8888, false)
                    aiVerifyExecutor.execute {
                        val verified = FaceMatcherEngine.verifyMatchWithGeminiVisionBlocking(
                            this@MainActivity,
                            frozenBmp,
                            matchedStudent
                        )
                        isOnlineAiVerifying = false
                        if (verified) {
                            mainHandler.post {
                                confirmAndMarkStudentAttendance(matchedStudent)
                            }
                        } else {
                            pendingCandidateId = null
                            pendingCandidateHits = 0
                            mainHandler.post {
                                updateBanner("⚠️ चेहरा मेल नहीं खाया, कृपया सीधे देखें", "#7F1D1D", "#F87171")
                            }
                        }
                    }
                    return
                } else if (!result.isHighConfidence) {
                    // यदि ऑफलाइन है और हाई-कॉन्फिडेंस नहीं है, तो कम से कम 3 लगातार फ्रेम की सख्त पुष्टि मांगें
                    if (pendingCandidateHits < 3) {
                        return
                    }
                }
            }

            // 100% पुष्टि होने पर हाज़िरी दर्ज करें
            pendingCandidateHits = 0
            mainHandler.post {
                confirmAndMarkStudentAttendance(matchedStudent)
            }
        } else {
            pendingCandidateId = null
            pendingCandidateHits = 0
            mainHandler.post {
                if (now - lastUnknownVoiceTimeMs > 6500L && !voiceManager.isCurrentlySpeaking) {
                    lastUnknownVoiceTimeMs = now
                    updateBanner("❓ चेहरा पहचाना नहीं गया, सीधे देखें", "#7F1D1D", "#F87171")
                }
            }
        }
    }

    private fun confirmAndMarkStudentAttendance(matched: Student) {
        val now = System.currentTimeMillis()
        val todayKey = AppStorage.getTodayShortKey()
        val todayList = currentPayload.attendance.getOrPut(todayKey) { mutableListOf() }
        val alreadyMarked = todayList.contains(matched.id)

        if (matched.id == lastRecognizedId && now - lastRecognizedTimeMs < 3800L) {
            samePersonStreak++
            if (samePersonStreak >= 3 && !voiceManager.isCurrentlySpeaking) {
                voiceManager.speakGreeting("move_away", matched.name)
                updateBanner("😊 ${matched.name}, अब अगले छात्र को आने दें!", "#854D0E", "#FACC15")
                samePersonStreak = 0
            }
            return
        }

        lastRecognizedId = matched.id
        lastRecognizedTimeMs = now
        samePersonStreak = 1

        if (!alreadyMarked) {
            todayList.add(matched.id)
            AppStorage.saveData(this, currentPayload)
            updateStatsBar()
            updateBanner("✅ ${matched.name} (${matched.className} - रोल ${matched.roll}) उपस्थित!", "#14532D", "#4ADE80")
            voiceManager.speakGreeting("success", matched.name)
        } else {
            updateBanner("ℹ️ ${matched.name} की हाज़िरी पहले ही लग चुकी है", "#1E3A8A", "#60A5FA")
            if (!voiceManager.isCurrentlySpeaking) {
                voiceManager.speakGreeting("warning", matched.name)
            }
        }
    }

    private fun updateBanner(msg: String, bgHex: String, borderHex: String) {
        statusBanner.text = msg
        statusBanner.background = GradientDrawable().apply {
            setColor(Color.parseColor(bgHex))
            cornerRadius = 32f
            setStroke(3, Color.parseColor(borderHex))
        }
    }
}
