package com.school.sarvodayaattendance

import android.Manifest
import android.annotation.SuppressLint
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Matrix
import android.net.Uri
import android.os.Bundle
import android.text.Editable
import android.text.InputFilter
import android.text.InputType
import android.text.TextWatcher
import android.view.View
import android.widget.Button
import android.widget.EditText
import android.widget.ImageButton
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.Camera
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import com.google.gson.Gson
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.face.FaceDetection
import com.google.mlkit.vision.face.FaceDetectorOptions
import com.school.sarvodayaattendance.data.AppStorage
import com.school.sarvodayaattendance.data.BackupPayload
import com.school.sarvodayaattendance.data.Student
import com.school.sarvodayaattendance.utils.FaceMatcherEngine
import com.school.sarvodayaattendance.utils.RealisticVoiceManager
import java.io.BufferedReader
import java.io.InputStreamReader
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class MainActivity : AppCompatActivity() {

    private lateinit var previewView: PreviewView
    private lateinit var tvHeaderDate: TextView
    private lateinit var tvScanStatusBadge: TextView
    private lateinit var btnOpenAdmin: Button
    private lateinit var btnManualTrigger: Button
    private lateinit var btnTorch: ImageButton
    private lateinit var btnFlipCamera: ImageButton
    private lateinit var btnToggleVoice: ImageButton
    private lateinit var btnShareTodayAttendance: Button
    private lateinit var btnImportPack: Button

    private lateinit var verifiedPopupCard: LinearLayout
    private lateinit var ivVerifiedPhoto: ImageView
    private lateinit var tvVerifiedName: TextView
    private lateinit var tvVerifiedSub: TextView

    private lateinit var cameraExecutor: ExecutorService
    private var camera: Camera? = null
    private var lensFacing = CameraSelector.LENS_FACING_FRONT
    private var isTorchOn = false
    private var isCooldownActive = false

    private lateinit var voiceManager: RealisticVoiceManager
    private lateinit var payload: BackupPayload

    private val faceDetector by lazy {
        val opts = FaceDetectorOptions.Builder()
            .setPerformanceMode(FaceDetectorOptions.PERFORMANCE_MODE_FAST)
            .setLandmarkMode(FaceDetectorOptions.LANDMARK_MODE_NONE)
            .setClassificationMode(FaceDetectorOptions.CLASSIFICATION_MODE_NONE)
            .setMinFaceSize(0.22f)
            .build()
        FaceDetection.getClient(opts)
    }

    private val cameraPermLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted ->
        if (granted) startCamera() else Toast.makeText(this, "कैमरा अनुमति आवश्यक है!", Toast.LENGTH_LONG).show()
    }

    private val fileImportLauncher = registerForActivityResult(
        ActivityResultContracts.GetContent()
    ) { uri: Uri? ->
        if (uri != null) importBackupFromFileUri(uri)
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        voiceManager = RealisticVoiceManager(this)
        cameraExecutor = Executors.newSingleThreadExecutor()

        bindViews()
        setupListeners()
    }

    override fun onResume() {
        super.onResume()
        payload = AppStorage.loadData(this)
        tvHeaderDate.text = AppStorage.getTodayFullDot()
        btnOpenAdmin.visibility = if (payload.isScannerOnlyPack) View.GONE else View.VISIBLE

        FaceMatcherEngine.prepareEmbeddings(payload.students)

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
            startCamera()
        } else {
            cameraPermLauncher.launch(Manifest.permission.CAMERA)
        }
    }

    private fun bindViews() {
        previewView = findViewById(R.id.previewView)
        tvHeaderDate = findViewById(R.id.tvHeaderDate)
        tvScanStatusBadge = findViewById(R.id.tvScanStatusBadge)
        btnOpenAdmin = findViewById(R.id.btnOpenAdmin)
        btnManualTrigger = findViewById(R.id.btnManualTrigger)
        btnTorch = findViewById(R.id.btnTorch)
        btnFlipCamera = findViewById(R.id.btnFlipCamera)
        btnToggleVoice = findViewById(R.id.btnToggleVoice)
        btnShareTodayAttendance = findViewById(R.id.btnShareTodayAttendance)
        btnImportPack = findViewById(R.id.btnImportPack)

        verifiedPopupCard = findViewById(R.id.verifiedPopupCard)
        ivVerifiedPhoto = findViewById(R.id.ivVerifiedPhoto)
        tvVerifiedName = findViewById(R.id.tvVerifiedName)
        tvVerifiedSub = findViewById(R.id.tvVerifiedSub)
    }

    private fun setupListeners() {
        btnOpenAdmin.setOnClickListener { showPinDialog() }

        btnManualTrigger.setOnClickListener {
            isCooldownActive = false
            tvScanStatusBadge.text = "🟢 चेहरा स्कैन किया जा रहा है..."
        }

        btnTorch.setOnClickListener {
            val cam = camera ?: return@setOnClickListener
            if (cam.cameraInfo.hasFlashUnit()) {
                isTorchOn = !isTorchOn
                cam.cameraControl.enableTorch(isTorchOn)
                btnTorch.alpha = if (isTorchOn) 1.0f else 0.6f
            } else {
                Toast.makeText(this, "फ्रंट कैमरे में फ्लैश उपलब्ध नहीं है", Toast.LENGTH_SHORT).show()
            }
        }

        btnFlipCamera.setOnClickListener {
            lensFacing = if (lensFacing == CameraSelector.LENS_FACING_FRONT) {
                CameraSelector.LENS_FACING_BACK
            } else {
                CameraSelector.LENS_FACING_FRONT
            }
            isTorchOn = false
            startCamera()
        }

        btnToggleVoice.setOnClickListener {
            voiceManager.isVoiceEnabled = !voiceManager.isVoiceEnabled
            btnToggleVoice.alpha = if (voiceManager.isVoiceEnabled) 1.0f else 0.4f
            val msg = if (voiceManager.isVoiceEnabled) "🔊 आवाज़ चालू" else "🔇 आवाज़ बंद"
            Toast.makeText(this, msg, Toast.LENGTH_SHORT).show()
        }

        btnShareTodayAttendance.setOnClickListener { shareTodayAttendanceJson() }

        btnImportPack.setOnClickListener { fileImportLauncher.launch("*/*") }
    }

    private fun startCamera() {
        val providerFuture = ProcessCameraProvider.getInstance(this)
        providerFuture.addListener({
            val provider = providerFuture.get()
            val preview = Preview.Builder().build().also {
                it.setSurfaceProvider(previewView.surfaceProvider)
            }

            val analyzer = ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()
                .also {
                    it.setAnalyzer(cameraExecutor) { imageProxy ->
                        processCameraFrame(imageProxy)
                    }
                }

            val selector = CameraSelector.Builder().requireLensFacing(lensFacing).build()
            try {
                provider.unbindAll()
                camera = provider.bindToLifecycle(this, selector, preview, analyzer)
            } catch (e: Exception) {
                e.printStackTrace()
            }
        }, ContextCompat.getMainExecutor(this))
    }

    @SuppressLint("UnsafeOptInUsageError")
    private fun processCameraFrame(imageProxy: ImageProxy) {
        if (isCooldownActive || payload.students.isEmpty()) {
            imageProxy.close()
            return
        }

        val mediaImage = imageProxy.image
        if (mediaImage == null) {
            imageProxy.close()
            return
        }

        val inputImage = InputImage.fromMediaImage(mediaImage, imageProxy.imageInfo.rotationDegrees)
        faceDetector.process(inputImage)
            .addOnSuccessListener { faces ->
                if (faces.isNotEmpty() && !isCooldownActive) {
                    val biggestFace = faces.maxByOrNull { it.boundingBox.width() * it.boundingBox.height() }
                    if (biggestFace != null && biggestFace.boundingBox.width() >= 60) {
                        val frameBmp = imageProxyToBitmap(imageProxy)
                        if (frameBmp != null) {
                            val match = FaceMatcherEngine.findBestMatch(
                                frameBmp,
                                biggestFace.boundingBox,
                                payload.students
                            )
                            if (match != null) {
                                isCooldownActive = true
                                runOnUiThread { handleMatchedStudent(match.first) }
                            }
                        }
                    }
                }
            }
            .addOnCompleteListener {
                imageProxy.close()
            }
    }

    private fun imageProxyToBitmap(imageProxy: ImageProxy): Bitmap? {
        return try {
            val bmp = imageProxy.toBitmap()
            val rot = imageProxy.imageInfo.rotationDegrees
            if (rot == 0) return bmp
            val matrix = Matrix().apply { postRotate(rot.toFloat()) }
            Bitmap.createBitmap(bmp, 0, 0, bmp.width, bmp.height, matrix, true)
        } catch (e: Exception) {
            null
        }
    }

    private fun handleMatchedStudent(student: Student) {
        val todayKey = AppStorage.getTodayShortKey()
        if (AppStorage.isSundayOrHoliday(payload, todayKey, student.className)) {
            val reason = AppStorage.getHolidayReason(payload, todayKey, student.className) ?: "अवकाश"
            Toast.makeText(this, "🏖️ आज $reason है!", Toast.LENGTH_SHORT).show()
            voiceManager.speakHindi("आज विद्यालय में अवकाश है।")
            resetCooldownDelayed(2500)
            return
        }

        val list = payload.attendance.getOrPut(todayKey) { mutableListOf() }
        val alreadyMarked = list.contains(student.id)
        if (!alreadyMarked) {
            list.add(student.id)
            AppStorage.saveData(this, payload)
        }

        val rollTwo = AppStorage.formatRoll(student.roll)
        tvVerifiedName.text = "${student.name} (रोल: $rollTwo)"
        tvVerifiedSub.text = if (alreadyMarked) {
            "कक्षा: ${student.className} | पहले से उपस्थित है"
        } else {
            "कक्षा: ${student.className} | उपस्थिति दर्ज हो गई"
        }

        val photoBmp = student.photos.firstOrNull()?.let { FaceMatcherEngine.decodeBase64Bitmap(it) }
        if (photoBmp != null) ivVerifiedPhoto.setImageBitmap(photoBmp)

        verifiedPopupCard.visibility = View.VISIBLE

        val speechText = if (alreadyMarked) {
            "${student.name}, रोल नंबर $rollTwo, आपकी हाज़िरी पहले ही लग चुकी है।"
        } else {
            "धन्यवाद, ${student.name}, रोल नंबर $rollTwo, कक्षा ${student.className}, आपकी उपस्थिति दर्ज हो गई।"
        }
        voiceManager.speakHindi(speechText)

        resetCooldownDelayed(2600)
    }

    private fun resetCooldownDelayed(ms: Long) {
        previewView.postDelayed({
            verifiedPopupCard.visibility = View.GONE
            isCooldownActive = false
        }, ms)
    }

    private fun showPinDialog() {
        val input = EditText(this).apply {
            inputType = InputType.TYPE_CLASS_NUMBER or InputType.TYPE_NUMBER_VARIATION_PASSWORD
            filters = arrayOf(InputFilter.LengthFilter(6))
            hint = "6-अंकों का पिन"
            textAlignment = View.TEXT_ALIGNMENT_CENTER
            textSize = 22f
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
        val ids = payload.attendance[todayKey] ?: emptyList<String>()
        if (ids.isEmpty()) {
            Toast.makeText(this, "आज अभी तक कोई उपस्थिति दर्ज नहीं हुई है!", Toast.LENGTH_SHORT).show()
            return
        }
        val exportMap = mapOf(
            "type" to "attendance_day_export",
            "dateKey" to todayKey,
            "presentStudentIds" to ids
        )
        val jsonStr = Gson().toJson(exportMap)
        val sendIntent = Intent(Intent.ACTION_SEND).apply {
            type = "text/plain"
            putExtra(Intent.EXTRA_SUBJECT, "Attendance_$todayKey.json")
            putExtra(Intent.EXTRA_TEXT, jsonStr)
        }
        startActivity(Intent.createChooser(sendIntent, "आज की हाज़िरी शेयर करें"))
    }

    private fun importBackupFromFileUri(uri: Uri) {
        try {
            val sb = StringBuilder()
            contentResolver.openInputStream(uri)?.use { inputStream ->
                BufferedReader(InputStreamReader(inputStream)).use { reader ->
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
                FaceMatcherEngine.prepareEmbeddings(payload.students)
                btnOpenAdmin.visibility = if (payload.isScannerOnlyPack) View.GONE else View.VISIBLE
                Toast.makeText(this, "✔ डेटा सफलतापूर्वक लोड हो गया! (${payload.students.size} छात्र)", Toast.LENGTH_LONG).show()
            } else {
                Toast.makeText(this, "❌ अमान्य फ़ाइल!", Toast.LENGTH_SHORT).show()
            }
        } catch (e: Exception) {
            Toast.makeText(this, "❌ फ़ाइल पढ़ने में त्रुटि!", Toast.LENGTH_SHORT).show()
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        voiceManager.shutdown()
        cameraExecutor.shutdown()
    }
}
