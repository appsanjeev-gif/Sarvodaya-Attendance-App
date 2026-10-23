package com.school.sarvodayaattendance

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.content.res.ColorStateList
import android.graphics.Bitmap
import android.graphics.Color
import android.net.Uri
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.text.Editable
import android.text.InputFilter
import android.text.InputType
import android.text.TextWatcher
import android.view.View
import android.widget.Button
import android.widget.EditText
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
import com.google.gson.Gson
import com.google.mlkit.vision.common.InputImage
import com.school.sarvodayaattendance.data.AppStorage
import com.school.sarvodayaattendance.data.BackupPayload
import com.school.sarvodayaattendance.data.Student
import com.school.sarvodayaattendance.utils.FaceMatcherEngine
import com.school.sarvodayaattendance.utils.RealisticVoiceManager
import java.io.BufferedReader
import java.io.InputStreamReader

class MainActivity : AppCompatActivity() {

    private lateinit var previewView: PreviewView
    private lateinit var tvHeaderDate: TextView
    private lateinit var tvLiveStatus: TextView
    private lateinit var tvScannerOnlyBanner: TextView
    private lateinit var btnAdminLock: Button
    private lateinit var btnTorchIcon: Button
    private lateinit var btnSwitchCamIcon: Button
    private lateinit var btnVoiceIcon: Button
    private lateinit var resultCardBox: LinearLayout
    private lateinit var tvResultLogo: TextView
    private lateinit var tvResultDate: TextView
    private lateinit var tvResultName: TextView
    private lateinit var tvResultMeta: TextView
    private lateinit var tvResultSubNote: TextView

    private lateinit var voiceManager: RealisticVoiceManager
    private lateinit var payload: BackupPayload

    private var camera: Camera? = null
    private var lensFacing = CameraSelector.LENS_FACING_FRONT
    private var isTorchOn = false
    private var isProcessingFrame = false
    private var lastScanTimestamp = 0L

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

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        voiceManager = RealisticVoiceManager(this)
        payload = AppStorage.loadData(this)

        previewView = findViewById(R.id.previewView)
        tvHeaderDate = findViewById(R.id.tvHeaderDate)
        tvLiveStatus = findViewById(R.id.tvLiveStatus)
        tvScannerOnlyBanner = findViewById(R.id.tvScannerOnlyBanner)
        btnAdminLock = findViewById(R.id.btnAdminLock)
        btnTorchIcon = findViewById(R.id.btnTorchIcon)
        btnSwitchCamIcon = findViewById(R.id.btnSwitchCamIcon)
        btnVoiceIcon = findViewById(R.id.btnVoiceIcon)
        resultCardBox = findViewById(R.id.resultCardBox)
        tvResultLogo = findViewById(R.id.tvResultLogo)
        tvResultDate = findViewById(R.id.tvResultDate)
        tvResultName = findViewById(R.id.tvResultName)
        tvResultMeta = findViewById(R.id.tvResultMeta)
        tvResultSubNote = findViewById(R.id.tvResultSubNote)

        tvHeaderDate.text = AppStorage.getTodayFullDot()
        refreshScannerModeUI()

        findViewById<Button>(R.id.btnManualScanTrigger).setOnClickListener {
            performLiveFrameCheck(true)
        }

        btnTorchIcon.setOnClickListener {
            isTorchOn = !isTorchOn
            camera?.cameraControl?.enableTorch(isTorchOn)
            btnTorchIcon.backgroundTintList = ColorStateList.valueOf(
                Color.parseColor(if (isTorchOn) "#16A34A" else "#334155")
            )
            Toast.makeText(this, if (isTorchOn) "🔦 टॉर्च चालू" else "🔦 टॉर्च बंद", Toast.LENGTH_SHORT).show()
        }

        btnSwitchCamIcon.setOnClickListener {
            lensFacing = if (lensFacing == CameraSelector.LENS_FACING_FRONT) {
                CameraSelector.LENS_FACING_BACK
            } else {
                CameraSelector.LENS_FACING_FRONT
            }
            startCamera()
            Toast.makeText(
                this,
                if (lensFacing == CameraSelector.LENS_FACING_FRONT) "🔄 फ्रंट कैमरा" else "🔄 बैक कैमरा",
                Toast.LENGTH_SHORT
            ).show()
        }

        btnVoiceIcon.setOnClickListener {
            voiceManager.isVoiceEnabled = !voiceManager.isVoiceEnabled
            btnVoiceIcon.text = if (voiceManager.isVoiceEnabled) "🔊" else "🔇"
            btnVoiceIcon.backgroundTintList = ColorStateList.valueOf(
                Color.parseColor(if (voiceManager.isVoiceEnabled) "#16A34A" else "#334155")
            )
        }

        btnAdminLock.setOnClickListener {
            if (payload.isScannerOnlyPack) {
                Toast.makeText(this, "🔒 यह सीमित स्कैनर पैक है!", Toast.LENGTH_SHORT).show()
            } else {
                showPinDialog()
            }
        }

        findViewById<Button>(R.id.btnShareTodayAtt).setOnClickListener {
            shareTodayAttendanceJson()
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
        payload = AppStorage.loadData(this)
        FaceMatcherEngine.clearCache()
        FaceMatcherEngine.warmUpCache(payload.students)
        refreshScannerModeUI()
        uiHandler.postDelayed(autoScanLoop, 800)
    }

    override fun onPause() {
        super.onPause()
        uiHandler.removeCallbacks(autoScanLoop)
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
        if (!isManualTap && (now - lastScanTimestamp < 2200)) return

        if (payload.students.isEmpty()) {
            if (isManualTap) {
                showResultBanner("error", null, "कोई छात्र पंजीकृत नहीं है!", 2000)
            }
            return
        }

        val frameBmp: Bitmap = previewView.bitmap ?: return
        isProcessingFrame = true

        val inputImage = InputImage.fromBitmap(frameBmp, 0)
        FaceMatcherEngine.detector.process(inputImage)
            .addOnSuccessListener { faces ->
                if (faces.isNotEmpty()) {
                    val primaryFace = faces.maxByOrNull { it.boundingBox.width() * it.boundingBox.height() }!!
                    val croppedFace = FaceMatcherEngine.cropFaceBitmap(frameBmp, primaryFace.boundingBox)
                    val (matchedStudent, score) = FaceMatcherEngine.findBestMatch(croppedFace, payload.students)

                    if (matchedStudent != null && score <= 19.0f) {
                        lastScanTimestamp = System.currentTimeMillis()
                        handleMatchedStudent(matchedStudent)
                    } else if (isManualTap) {
                        lastScanTimestamp = System.currentTimeMillis()
                        showResultBanner("error", null, "चेहरा मैच नहीं हुआ!", 2000)
                        voiceManager.speakGreeting("error", "")
                    }
                } else if (isManualTap) {
                    showResultBanner("error", null, "कैमरे में चेहरा साफ़ नहीं दिखा!", 2000)
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
            Toast.makeText(this, "⚠️ आज ${stu.className} के लिए अवकाश है!", Toast.LENGTH_SHORT).show()
            return
        }

        val list = payload.attendance.getOrPut(todayKey) { mutableListOf() }
        if (!list.contains(stu.id)) {
            list.add(stu.id)
            AppStorage.saveData(this, payload)
            showResultBanner("success", stu, "✔ हाज़िरी दर्ज हुई", 4000)
            voiceManager.speakGreeting("success", stu.name)
        } else {
            showResultBanner("warning", stu, "⚠️ आज की हाज़िरी पहले से दर्ज है!", 3000)
            voiceManager.speakGreeting("warning", stu.name)
        }
    }

    private fun showResultBanner(type: String, stu: Student?, note: String, durationMs: Long) {
        tvResultDate.text = AppStorage.getTodayFullDot()
        when (type) {
            "success" -> {
                tvResultLogo.text = "✅"
                tvResultName.text = "नाम: ${stu?.name ?: ""}"
                tvResultMeta.text = "कक्षा: ${stu?.className ?: ""} | रोल नं: ${AppStorage.formatRoll(stu?.roll ?: "")}"
                tvResultSubNote.setTextColor(Color.parseColor("#15803D"))
                tvResultSubNote.text = note
            }
            "warning" -> {
                tvResultLogo.text = "⚠️"
                tvResultName.text = "नाम: ${stu?.name ?: ""}"
                tvResultMeta.text = "कक्षा: ${stu?.className ?: ""} | रोल नं: ${AppStorage.formatRoll(stu?.roll ?: "")}"
                tvResultSubNote.setTextColor(Color.parseColor("#B45309"))
                tvResultSubNote.text = note
            }
            else -> {
                tvResultLogo.text = "❌"
                tvResultName.text = "चेहरा मैच नहीं हुआ!"
                tvResultMeta.text = note
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

    private fun showPinDialog() {
        val input = EditText(this).apply {
            inputType = InputType.TYPE_CLASS_NUMBER or InputType.TYPE_NUMBER_VARIATION_PASSWORD
            filters = arrayOf(InputFilter.LengthFilter(6))
            hint = "6-अंकों का पिन (डिफ़ॉल्ट: 123456)"
            textAlignment = View.TEXT_ALIGNMENT_CENTER
            textSize = 20f
            setPadding(45, 35, 45, 35)
        }
        val dialog = AlertDialog.Builder(this)
            .setTitle("🔐 एडमिन सुरक्षा पिन")
            .setView(input)
            .setPositiveButton("खोलें") { _, _ ->
                if (input.text.toString().trim() == payload.adminPin) {
                    startActivity(Intent(this, AdminActivity::class.java))
                } else {
                    Toast.makeText(this, "❌ गलत पिन!", Toast.LENGTH_SHORT).show()
                }
            }
            .setNegativeButton("रद्द", null)
            .create()

        input.addTextChangedListener(object : TextWatcher {
            override fun beforeTextChanged(s: CharSequence?, start: Int, count: Int, after: Int) {}
            override fun onTextChanged(s: CharSequence?, start: Int, before: Int, count: Int) {}
            override fun afterTextChanged(s: Editable?) {
                if (s != null && s.length == 6) {
                    if (s.toString() == payload.adminPin) {
                        dialog.dismiss()
                        startActivity(Intent(this@MainActivity, AdminActivity::class.java))
                    } else {
                        Toast.makeText(this@MainActivity, "❌ गलत पिन!", Toast.LENGTH_SHORT).show()
                        s.clear()
                    }
                }
            }
        })
        dialog.show()
    }

    private fun shareTodayAttendanceJson() {
        val todayKey = AppStorage.getTodayShortKey()
        val presentIds = payload.attendance[todayKey] ?: emptyList()
        if (presentIds.isEmpty()) {
            Toast.makeText(this, "⚠️ आज अभी तक किसी की हाज़िरी नहीं लगी है!", Toast.LENGTH_SHORT).show()
            return
        }
        val exportMap = mapOf(
            "type" to "DAILY_ATTENDANCE_SYNC",
            "dateKey" to todayKey,
            "presentStudentIds" to presentIds
        )
        val json = Gson().toJson(exportMap)
        val sendIntent = Intent(Intent.ACTION_SEND).apply {
            type = "text/plain"
            putExtra(Intent.EXTRA_SUBJECT, "Today_Attendance_$todayKey.json")
            putExtra(Intent.EXTRA_TEXT, json)
        }
        startActivity(Intent.createChooser(sendIntent, "आज की हाज़िरी भेजें"))
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
                Toast.makeText(this, "✔ डेटा सफलतापूर्वक लोड हो गया! (${payload.students.size} छात्र)", Toast.LENGTH_LONG).show()
            } else {
                Toast.makeText(this, "❌ अमान्य फ़ाइल!", Toast.LENGTH_SHORT).show()
            }
        } catch (e: Exception) {
            Toast.makeText(this, "❌ फ़ाइल पढ़ने में त्रुटि!", Toast.LENGTH_SHORT).show()
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
