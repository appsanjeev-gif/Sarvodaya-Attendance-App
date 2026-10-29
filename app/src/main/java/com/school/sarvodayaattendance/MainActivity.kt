package com.school.sarvodayaattendance

import android.Manifest
import android.animation.ObjectAnimator
import android.animation.ValueAnimator
import android.annotation.SuppressLint
import android.content.ClipData
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.content.res.ColorStateList
import android.graphics.Bitmap
import android.graphics.Color
import android.graphics.Typeface
import android.graphics.drawable.GradientDrawable
import android.net.Uri
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.text.Editable
import android.text.InputFilter
import android.text.InputType
import android.text.TextWatcher
import android.view.Gravity
import android.view.MotionEvent
import android.view.View
import android.view.WindowManager
import android.view.inputmethod.InputMethodManager
import android.widget.Button
import android.widget.EditText
import android.widget.FrameLayout
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.Camera
import androidx.camera.core.CameraSelector
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.core.content.FileProvider
import com.google.gson.Gson
import com.google.mlkit.vision.common.InputImage
import com.school.sarvodayaattendance.data.AppStorage
import com.school.sarvodayaattendance.data.BackupPayload
import com.school.sarvodayaattendance.data.Student
import com.school.sarvodayaattendance.utils.FaceMatcherEngine
import com.school.sarvodayaattendance.utils.RealisticVoiceManager
import java.io.BufferedReader
import java.io.File
import java.io.FileOutputStream
import java.io.InputStreamReader
import kotlin.math.roundToInt

class MainActivity : AppCompatActivity() {

    private lateinit var previewView: PreviewView
    private lateinit var tvHeaderDate: TextView
    private lateinit var tvLiveStatus: TextView
    private lateinit var tvLiveTickerCount: TextView
    private lateinit var tvScannerOnlyBanner: TextView
    private lateinit var btnAdminLock: Button
    private lateinit var btnTorchIcon: Button
    private lateinit var btnSwitchCamIcon: Button
    private lateinit var btnVoiceIcon: Button
    private lateinit var resultCardBox: LinearLayout
    private lateinit var ivResultStudentPhoto: ImageView
    private lateinit var tvResultLogo: TextView
    private lateinit var tvResultDate: TextView
    private lateinit var tvResultName: TextView
    private lateinit var tvResultMeta: TextView
    private lateinit var tvResultSubNote: TextView
    private lateinit var logoVipTrigger: FrameLayout

    private lateinit var voiceManager: RealisticVoiceManager
    private lateinit var payload: BackupPayload

    private var camera: Camera? = null
    private var lensFacing = CameraSelector.LENS_FACING_FRONT
    private var isTorchOn = false
    private var isProcessingFrame = false
    private var lastScanTimestamp = 0L

    private var isVipInspectionModeActive = false
    private var hasSpokenVipForCurrentFace = false
    private var vipHoldRunnable: Runnable? = null

    private var lastSpokenStudentId: String? = null
    private var sameStudentScanStreak: Int = 0
    private var emptyFrameCount: Int = 0

    private val uiHandler = Handler(Looper.getMainLooper())
    private var hideCardRunnable: Runnable? = null

    private val packFilePicker = registerForActivityResult(ActivityResultContracts.GetContent()) { uri: Uri? ->
        if (uri != null) {
            importBackupFromUri(uri)
        }
    }

    private val autoScanLoop = object : Runnable {
        override fun run() {
            performLiveFrameCheck(false)
            uiHandler.postDelayed(this, 650)
        }
    }

    @SuppressLint("ClickableViewAccessibility")
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        FaceMatcherEngine.initEngine(this)

        voiceManager = RealisticVoiceManager(this)
        uiHandler.postDelayed({ voiceManager.speakStartupWelcome() }, 600)

        payload = AppStorage.loadData(this)

        previewView = findViewById(R.id.previewView)
        tvHeaderDate = findViewById(R.id.tvHeaderDate)
        tvLiveStatus = findViewById(R.id.tvLiveStatus)
        tvLiveTickerCount = findViewById(R.id.tvLiveTickerCount)
        tvScannerOnlyBanner = findViewById(R.id.tvScannerOnlyBanner)
        btnAdminLock = findViewById(R.id.btnAdminLock)
        btnTorchIcon = findViewById(R.id.btnTorchIcon)
        btnSwitchCamIcon = findViewById(R.id.btnSwitchCamIcon)
        btnVoiceIcon = findViewById(R.id.btnVoiceIcon)
        resultCardBox = findViewById(R.id.resultCardBox)
        ivResultStudentPhoto = findViewById(R.id.ivResultStudentPhoto)
        tvResultLogo = findViewById(R.id.tvResultLogo)
        tvResultDate = findViewById(R.id.tvResultDate)
        tvResultName = findViewById(R.id.tvResultName)
        tvResultMeta = findViewById(R.id.tvResultMeta)
        tvResultSubNote = findViewById(R.id.tvResultSubNote)
        logoVipTrigger = findViewById(R.id.logoVipTrigger)

        tvHeaderDate.text = AppStorage.getTodayFullDot()

        findViewById<View>(R.id.laserScanLine)?.let { laser ->
            ObjectAnimator.ofFloat(laser, "translationY", 0f, 190f).apply {
                duration = 1400
                repeatMode = ValueAnimator.REVERSE
                repeatCount = ValueAnimator.INFINITE
                start()
            }
        }

        logoVipTrigger.setOnTouchListener { _, event ->
            when (event.action) {
                MotionEvent.ACTION_DOWN -> {
                    vipHoldRunnable?.let { uiHandler.removeCallbacks(it) }
                    vipHoldRunnable = Runnable {
                        toggleVipInspectionModeSilent()
                    }
                    uiHandler.postDelayed(vipHoldRunnable!!, 2000L)
                    true
                }
                MotionEvent.ACTION_UP, MotionEvent.ACTION_CANCEL -> {
                    vipHoldRunnable?.let { uiHandler.removeCallbacks(it) }
                    true
                }
                else -> false
            }
        }

        refreshScannerModeUI()
        updateLiveSchoolTicker()

        findViewById<Button>(R.id.btnManualScanTrigger).setOnClickListener {
            performLiveFrameCheck(true)
        }

        btnTorchIcon.setOnClickListener {
            isTorchOn = !isTorchOn
            camera?.cameraControl?.enableTorch(isTorchOn)
            btnTorchIcon.backgroundTintList = ColorStateList.valueOf(
                Color.parseColor(if (isTorchOn) "#16A34A" else "#334155")
            )
        }

        btnSwitchCamIcon.setOnClickListener {
            lensFacing = if (lensFacing == CameraSelector.LENS_FACING_FRONT) {
                CameraSelector.LENS_FACING_BACK
            } else {
                CameraSelector.LENS_FACING_FRONT
            }
            startCamera()
        }

        btnVoiceIcon.setOnClickListener {
            voiceManager.isVoiceEnabled = !voiceManager.isVoiceEnabled
            btnVoiceIcon.text = if (voiceManager.isVoiceEnabled) "🔊" else "🔇"
            btnVoiceIcon.backgroundTintList = ColorStateList.valueOf(
                Color.parseColor(if (voiceManager.isVoiceEnabled) "#16A34A" else "#334155")
            )
        }

        btnVoiceIcon.setOnLongClickListener {
            val keyInput = EditText(this).apply {
                setText(voiceManager.getApiKey())
                hint = "Key1, Key2, Key3..."
                textSize = 15f
                minLines = 3
                maxLines = 8
                setPadding(40, 35, 40, 35)
            }
            AlertDialog.Builder(this)
                .setTitle("🎙️ Multi-API Keys (4-15)")
                .setView(keyInput)
                .setPositiveButton("सेव") { _, _ ->
                    val newKeys = keyInput.text.toString().trim()
                    voiceManager.saveCustomApiKey(newKeys)
                    Toast.makeText(this, "✔ सेव हो गया", Toast.LENGTH_SHORT).show()
                }
                .show()
            true
        }

        btnAdminLock.setOnClickListener {
            if (!payload.isScannerOnlyPack) {
                showRoyalPinDialog()
            }
        }

        findViewById<Button>(R.id.btnShareTodayAtt).setOnClickListener {
            shareTodayAttendanceDirect()
        }

        findViewById<Button>(R.id.btnLoadPack).setOnClickListener {
            packFilePicker.launch("*/*")
        }

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
            startCamera()
        } else {
            ActivityCompat.requestPermissions(this, arrayOf(Manifest.permission.CAMERA), 101)
        }
    }

    override fun onResume() {
        super.onResume()
        FaceMatcherEngine.initEngine(this)
        payload = AppStorage.loadData(this)
        FaceMatcherEngine.clearCache()
        FaceMatcherEngine.warmUpCache(payload.students)
        refreshScannerModeUI()
        updateLiveSchoolTicker()
        uiHandler.postDelayed(autoScanLoop, 800)
    }

    override fun onPause() {
        super.onPause()
        uiHandler.removeCallbacks(autoScanLoop)
    }

    private fun updateLiveSchoolTicker() {
        val todayKey = AppStorage.getTodayShortKey()
        val total = payload.students.size
        val presentIds = payload.attendance[todayKey] ?: emptyList()
        val present = payload.students.count { presentIds.contains(it.id) }
        val pct = if (total > 0) ((present * 100.0) / total).roundToInt() else 0
        tvLiveTickerCount.text = "📊 उपस्थित: $present/$total ($pct%)"
    }

    private fun toggleVipInspectionModeSilent() {
        isVipInspectionModeActive = !isVipInspectionModeActive
        hasSpokenVipForCurrentFace = false

        if (isVipInspectionModeActive) {
            logoVipTrigger.setBackgroundResource(R.drawable.bg_vip_glow)
        } else {
            logoVipTrigger.background = null
            resultCardBox.visibility = View.GONE
        }
    }

    // ✅ VIP स्वागत कार्ड जिसमें अब सही "स.म.वि." वाला लोगो दिखेगा और असली AI आवाज़ बोलेगी[cite: 12]
    private fun speakAndShowVipWelcomeOnFaceDetected() {
        val todayKey = AppStorage.getTodayShortKey()
        val total = payload.students.size
        val presentIds = payload.attendance[todayKey] ?: emptyList()
        val present = payload.students.count { presentIds.contains(it.id) }
        val pct = if (total > 0) ((present * 100.0) / total).roundToInt() else 0

        ivResultStudentPhoto.visibility = View.VISIBLE
        ivResultStudentPhoto.setImageResource(R.drawable.ic_smv_real_logo)
        tvResultLogo.text = "✨"
        tvResultDate.text = AppStorage.getTodayFullDot()
        tvResultName.text = "माननीय अधिकारी एवं अतिथि महोदय"
        tvResultMeta.text = "हार्दिक अभिनंदन एवं स्वागत"
        tvResultSubNote.setTextColor(Color.parseColor("#1E3A8A"))
        tvResultSubNote.text = "आज की लाइव उपस्थिति: $present / $total ($pct%)"

        resultCardBox.visibility = View.VISIBLE
        hideCardRunnable?.let { uiHandler.removeCallbacks(it) }
        hideCardRunnable = Runnable { resultCardBox.visibility = View.GONE }
        uiHandler.postDelayed(hideCardRunnable!!, 8000)

        voiceManager.speakVipInspectionWelcome(present, total)
    }

    private fun refreshScannerModeUI() {
        if (payload.isScannerOnlyPack) {
            tvScannerOnlyBanner.visibility = View.VISIBLE
            btnAdminLock.visibility = View.GONE
        } else {
            tvScannerOnlyBanner.visibility = View.GONE
            btnAdminLock.visibility = View.VISIBLE
        }
    }

    private fun startCamera() {
        val providerFuture = ProcessCameraProvider.getInstance(this)
        providerFuture.addListener({
            try {
                val provider = providerFuture.get()
                val preview = Preview.Builder().build().also {
                    it.setSurfaceProvider(previewView.surfaceProvider)
                }
                val selector = CameraSelector.Builder().requireLensFacing(lensFacing).build()
                provider.unbindAll()
                camera = provider.bindToLifecycle(this, selector, preview)
            } catch (e: Exception) {
                e.printStackTrace()
            }
        }, ContextCompat.getMainExecutor(this))
    }

    private fun performLiveFrameCheck(isManualTap: Boolean) {
        if (isProcessingFrame) return
        val now = System.currentTimeMillis()
        if (!isManualTap && (now - lastScanTimestamp < 1600)) return

        val frameBmp: Bitmap = previewView.bitmap ?: return
        isProcessingFrame = true

        val inputImage = InputImage.fromBitmap(frameBmp, 0)
        FaceMatcherEngine.detector.process(inputImage)
            .addOnSuccessListener { faces ->
                if (faces.isNotEmpty()) {
                    emptyFrameCount = 0
                    val primaryFace = faces.maxByOrNull { it.boundingBox.width() * it.boundingBox.height() }!!
                    val croppedFace = FaceMatcherEngine.cropFaceBitmap(frameBmp, primaryFace.boundingBox)
                    val (matchedStudent, score) = FaceMatcherEngine.findBestMatch(croppedFace, payload.students)

                    if (matchedStudent != null && score <= 19.0f) {
                        if (!isManualTap && voiceManager.isCurrentlySpeaking && matchedStudent.id == lastSpokenStudentId) {
                            isProcessingFrame = false
                            return@addOnSuccessListener
                        }
                        if (!isManualTap && matchedStudent.id == lastSpokenStudentId && (now - lastScanTimestamp < 4200)) {
                            isProcessingFrame = false
                            return@addOnSuccessListener
                        }

                        lastScanTimestamp = System.currentTimeMillis()
                        handleMatchedStudent(matchedStudent)
                    } else if (isVipInspectionModeActive) {
                        if (!hasSpokenVipForCurrentFace && !voiceManager.isCurrentlySpeaking) {
                            hasSpokenVipForCurrentFace = true
                            lastScanTimestamp = System.currentTimeMillis()
                            speakAndShowVipWelcomeOnFaceDetected()
                        }
                    } else if (isManualTap) {
                        lastScanTimestamp = System.currentTimeMillis()
                        if (payload.students.isEmpty()) {
                            showResultBanner("error", null, "कोई छात्र पंजीकृत नहीं है!", 2000)
                        } else {
                            showResultBanner("error", null, "चेहरा मैच नहीं हुआ!", 2000)
                            voiceManager.speakGreeting("error", "")
                        }
                    }
                } else {
                    emptyFrameCount++
                    if (emptyFrameCount >= 5) {
                        lastSpokenStudentId = null
                        sameStudentScanStreak = 0
                        hasSpokenVipForCurrentFace = false
                    }
                    if (isManualTap) {
                        showResultBanner("error", null, "कैमरे में चेहरा नहीं दिखा!", 2000)
                    }
                }
                isProcessingFrame = false
            }
            .addOnFailureListener {
                isProcessingFrame = false
            }
    }

    private fun handleMatchedStudent(stu: Student) {
        val todayKey = AppStorage.getTodayShortKey()
        if (AppStorage.isSundayOrHoliday(payload, todayKey, stu.className)) {
            Toast.makeText(this, "🏖️ अवकाश", Toast.LENGTH_SHORT).show()
            return
        }

        if (lastSpokenStudentId == stu.id) {
            sameStudentScanStreak++
        } else {
            lastSpokenStudentId = stu.id
            sameStudentScanStreak = 1
        }

        val list = payload.attendance.getOrPut(todayKey) { mutableListOf() }
        if (!list.contains(stu.id)) {
            list.add(stu.id)
            AppStorage.saveData(this, payload)
            updateLiveSchoolTicker()
            showResultBanner("success", stu, "✔ उपस्थित दर्ज", 4000)
            voiceManager.speakGreeting("success", stu.name)
        } else {
            if (sameStudentScanStreak >= 3) {
                val firstName = stu.name.trim().substringBefore(" ")
                showResultBanner("warning", stu, "😊 चलो अब आगे बढ़ो $firstName!", 4200)
                voiceManager.speakGreeting("move_away", stu.name)
            } else {
                showResultBanner("warning", stu, "⚠️ पहले से दर्ज है", 3500)
                voiceManager.speakGreeting("warning", stu.name)
            }
        }
    }

    private fun showResultBanner(type: String, stu: Student?, note: String, durationMs: Long) {
        tvResultDate.text = AppStorage.getTodayFullDot()

        if (stu != null && stu.photos.isNotEmpty()) {
            val bmp = FaceMatcherEngine.decodeBase64Bitmap(stu.photos[0])
            if (bmp != null) {
                ivResultStudentPhoto.setImageBitmap(bmp)
                ivResultStudentPhoto.visibility = View.VISIBLE
            } else {
                ivResultStudentPhoto.visibility = View.GONE
            }
        } else {
            ivResultStudentPhoto.visibility = View.GONE
        }

        when (type) {
            "success" -> {
                tvResultLogo.text = "✅"
                tvResultName.text = stu?.name ?: ""
                tvResultMeta.text = "कक्षा: ${stu?.className ?: ""} | रोल: ${AppStorage.formatRoll(stu?.roll ?: "")}"
                tvResultSubNote.setTextColor(Color.parseColor("#15803D"))
                tvResultSubNote.text = note
            }
            "warning" -> {
                tvResultLogo.text = "⚠️"
                tvResultName.text = stu?.name ?: ""
                tvResultMeta.text = "कक्षा: ${stu?.className ?: ""} | रोल: ${AppStorage.formatRoll(stu?.roll ?: "")}"
                tvResultSubNote.setTextColor(Color.parseColor("#B45309"))
                tvResultSubNote.text = note
            }
            else -> {
                tvResultLogo.text = "❌"
                tvResultName.text = note
                tvResultMeta.text = ""
                tvResultSubNote.text = ""
            }
        }

        resultCardBox.visibility = View.VISIBLE
        hideCardRunnable?.let { uiHandler.removeCallbacks(it) }
        hideCardRunnable = Runnable {
            resultCardBox.visibility = View.GONE
        }
        uiHandler.postDelayed(hideCardRunnable!!, durationMs)
    }

    private fun showRoyalPinDialog() {
        val container = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            gravity = Gravity.CENTER
            setPadding(55, 50, 55, 55)
            background = GradientDrawable().apply {
                setColor(Color.parseColor("#0F172A"))
                cornerRadius = 42f
                setStroke(3, Color.parseColor("#38BDF8"))
            }
        }

        val titleTv = TextView(this).apply {
            text = "🔐 6 अंक पिन"
            textSize = 20f
            setTypeface(null, Typeface.BOLD)
            setTextColor(Color.WHITE)
            gravity = Gravity.CENTER
            setPadding(0, 0, 0, 30)
        }

        val input = EditText(this).apply {
            inputType = InputType.TYPE_CLASS_NUMBER or InputType.TYPE_NUMBER_VARIATION_PASSWORD
            filters = arrayOf(InputFilter.LengthFilter(6))
            hint = "• • • • • •"
            setHintTextColor(Color.parseColor("#64748B"))
            setTextColor(Color.parseColor("#38BDF8"))
            textAlignment = View.TEXT_ALIGNMENT_CENTER
            textSize = 28f
            setTypeface(null, Typeface.BOLD)
            setPadding(30, 24, 30, 24)
            background = GradientDrawable().apply {
                setColor(Color.parseColor("#1E293B"))
                cornerRadius = 24f
                setStroke(2, Color.parseColor("#475569"))
            }
        }

        container.addView(titleTv)
        container.addView(input)

        val dialog = AlertDialog.Builder(this)
            .setView(container)
            .setCancelable(true)
            .create()

        dialog.window?.setBackgroundDrawableResource(android.R.color.transparent)
        dialog.window?.setSoftInputMode(WindowManager.LayoutParams.SOFT_INPUT_STATE_ALWAYS_VISIBLE)

        input.addTextChangedListener(object : TextWatcher {
            override fun beforeTextChanged(s: CharSequence?, start: Int, count: Int, after: Int) {}
            override fun onTextChanged(s: CharSequence?, start: Int, before: Int, count: Int) {}
            override fun afterTextChanged(s: Editable?) {
                if (s != null && s.length == 6) {
                    if (s.toString() == payload.adminPin) {
                        dialog.dismiss()
                        startActivity(Intent(this@MainActivity, AdminActivity::class.java))
                    } else {
                        Toast.makeText(this@MainActivity, "❌ गलत पिन", Toast.LENGTH_SHORT).show()
                        s.clear()
                    }
                }
            }
        })

        dialog.show()
        input.requestFocus()
        input.postDelayed({
            val imm = getSystemService(Context.INPUT_METHOD_SERVICE) as InputMethodManager
            imm.showSoftInput(input, InputMethodManager.SHOW_IMPLICIT)
        }, 120)
    }

    private fun shareTodayAttendanceDirect() {
        val todayKey = AppStorage.getTodayShortKey()
        val presentIds = payload.attendance[todayKey] ?: emptyList()
        if (presentIds.isEmpty()) {
            Toast.makeText(this, "⚠️ आज कोई उपस्थिति नहीं है", Toast.LENGTH_SHORT).show()
            return
        }
        val exportMap = mapOf(
            "type" to "DAILY_ATTENDANCE_SYNC",
            "dateKey" to todayKey,
            "presentStudentIds" to presentIds
        )
        val json = Gson().toJson(exportMap)
        try {
            val fileName = "Attendance_${todayKey.replace(".", "_")}.txt"
            val outFile = File(cacheDir, fileName)
            FileOutputStream(outFile).use { fos ->
                fos.write(json.toByteArray(Charsets.UTF_8))
            }
            val contentUri = FileProvider.getUriForFile(
                this,
                "${applicationContext.packageName}.fileprovider",
                outFile
            )
            val sendIntent = Intent(Intent.ACTION_SEND).apply {
                type = "text/plain"
                putExtra(Intent.EXTRA_STREAM, contentUri)
                clipData = ClipData.newRawUri(fileName, contentUri)
                addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
            }
            startActivity(Intent.createChooser(sendIntent, "हाज़िरी भेजें"))
        } catch (e: Exception) {
            Toast.makeText(this, "❌ शेयर त्रुटि", Toast.LENGTH_SHORT).show()
        }
    }

    private fun importBackupFromUri(uri: Uri) {
        try {
            val sb = StringBuilder()
            contentResolver.openInputStream(uri)?.use { stream ->
                BufferedReader(InputStreamReader(stream)).use { reader ->
                    var line: String?
                    while (reader.readLine().also { line = it } != null) {
                        sb.append(line).append("\n")
                    }
                }
            }
            val ok = AppStorage.importFromHtmlOrJson(this, sb.toString())
            if (ok) {
                payload = AppStorage.loadData(this)
                FaceMatcherEngine.clearCache()
                FaceMatcherEngine.warmUpCache(payload.students)
                refreshScannerModeUI()
                updateLiveSchoolTicker()
                Toast.makeText(this, "✔ पैक लोड हो गया (${payload.students.size} छात्र)", Toast.LENGTH_SHORT).show()
            } else {
                Toast.makeText(this, "❌ अमान्य फ़ाइल", Toast.LENGTH_SHORT).show()
            }
        } catch (e: Exception) {
            Toast.makeText(this, "❌ त्रुटि", Toast.LENGTH_SHORT).show()
        }
    }

    override fun onRequestPermissionsResult(requestCode: Int, permissions: Array<out String>, grantResults: IntArray) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == 101 && grantResults.isNotEmpty() && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
            startCamera()
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        voiceManager.shutdown()
    }
}
