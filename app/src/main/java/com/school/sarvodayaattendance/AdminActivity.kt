package com.school.sarvodayaattendance

import android.Manifest
import android.annotation.SuppressLint
import android.content.ClipData
import android.content.ContentValues
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Color
import android.graphics.Matrix
import android.graphics.Typeface
import android.graphics.drawable.GradientDrawable
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.os.Environment
import android.print.PrintAttributes
import android.print.PrintManager
import android.provider.MediaStore
import android.util.Base64
import android.view.Gravity
import android.view.ViewGroup
import android.webkit.JavascriptInterface
import android.webkit.WebChromeClient
import android.webkit.WebSettings
import android.webkit.WebView
import android.webkit.WebViewClient
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
import androidx.camera.core.CameraSelector
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.core.content.FileProvider
import com.google.gson.Gson
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.face.Face
import com.school.sarvodayaattendance.data.AppStorage
import com.school.sarvodayaattendance.data.BackupPayload
import com.school.sarvodayaattendance.utils.FaceMatcherEngine
import com.school.sarvodayaattendance.utils.RealisticVoiceManager
import java.io.BufferedReader
import java.io.ByteArrayOutputStream
import java.io.File
import java.io.FileOutputStream
import java.io.InputStreamReader

class AdminActivity : AppCompatActivity() {

    private lateinit var webView: WebView
    private var printWebView: WebView? = null
    private val gson = Gson()

    private var kycOverlayRoot: FrameLayout? = null
    private var kycPreviewView: PreviewView? = null
    private var kycLensFacing = CameraSelector.LENS_FACING_BACK
    private var pendingRestoreMode = "RESTORE"

    private val filePickerLauncher = registerForActivityResult(ActivityResultContracts.GetContent()) { uri: Uri? ->
        if (uri != null) {
            handlePickedFileUri(uri)
        }
    }

    @SuppressLint("SetJavaScriptEnabled")
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        FaceMatcherEngine.initEngine(this)

        val rootContainer = FrameLayout(this)
        webView = WebView(this)
        rootContainer.addView(
            webView,
            FrameLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.MATCH_PARENT
            )
        )
        setContentView(rootContainer)

        webView.settings.apply {
            javaScriptEnabled = true
            domStorageEnabled = true
            allowFileAccess = true
            allowContentAccess = true
            mixedContentMode = WebSettings.MIXED_CONTENT_ALWAYS_ALLOW
        }

        webView.webChromeClient = WebChromeClient()
        webView.webViewClient = WebViewClient()
        webView.addJavascriptInterface(AdminWebBridge(), "AndroidBridge")
        webView.loadUrl("file:///android_asset/admin.html")
    }

    private fun handlePickedFileUri(uri: Uri) {
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
            val rawContent = sb.toString()

            if (pendingRestoreMode == "RESTORE") {
                val ok = AppStorage.importFromHtmlOrJson(this, rawContent)
                if (ok) {
                    val loaded = AppStorage.loadData(this)
                    loaded.isScannerOnlyPack = false
                    AppStorage.saveData(this, loaded)
                    FaceMatcherEngine.clearCache()
                    FaceMatcherEngine.warmUpCache(loaded.students)
                    Toast.makeText(this, "✔ डेटा रिस्टोर हो गया (${loaded.students.size} छात्र)", Toast.LENGTH_SHORT).show()
                    webView.reload()
                } else {
                    Toast.makeText(this, "❌ अमान्य बैकअप फ़ाइल", Toast.LENGTH_SHORT).show()
                }
            } else if (pendingRestoreMode == "MERGE") {
                val escaped = gson.toJson(rawContent)
                webView.evaluateJavascript("onNativeAttendanceMergeContent($escaped);", null)
            }
        } catch (e: Exception) {
            Toast.makeText(this, "❌ फ़ाइल पढ़ने में त्रुटि", Toast.LENGTH_SHORT).show()
        }
    }

    private fun showLiveKycCameraOverlay(rootContainer: FrameLayout) {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this, arrayOf(Manifest.permission.CAMERA), 202)
            return
        }

        kycOverlayRoot?.let { rootContainer.removeView(it) }

        val overlay = FrameLayout(this).apply {
            setBackgroundColor(0xFF0F172A.toInt())
            isClickable = true
            isFocusable = true
        }

        val preview = PreviewView(this)
        kycPreviewView = preview
        overlay.addView(
            preview,
            FrameLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.MATCH_PARENT
            )
        )

        val bottomBar = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER
            setPadding(30, 30, 30, 50)
            setBackgroundColor(0x99000000.toInt())
        }

        val btnCancel = Button(this).apply {
            text = "✕ बंद"
            setTextColor(0xFFFFFFFF.toInt())
            setBackgroundColor(0xFFDC2626.toInt())
            setOnClickListener {
                rootContainer.removeView(overlay)
                kycOverlayRoot = null
            }
        }

        val btnSnap = Button(this).apply {
            text = "📸 फोटो खींचें"
            textSize = 16f
            setTextColor(0xFFFFFFFF.toInt())
            setBackgroundColor(0xFF16A34A.toInt())
            val lp = LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f)
            lp.setMargins(24, 0, 24, 0)
            layoutParams = lp
            setOnClickListener {
                captureAndVerifyKycFace(rootContainer, overlay)
            }
        }

        val btnFlip = Button(this).apply {
            text = "🔄 पलटें"
            setTextColor(0xFFFFFFFF.toInt())
            setBackgroundColor(0xFF334155.toInt())
            setOnClickListener {
                kycLensFacing = if (kycLensFacing == CameraSelector.LENS_FACING_FRONT) {
                    CameraSelector.LENS_FACING_BACK
                } else {
                    CameraSelector.LENS_FACING_FRONT
                }
                bindKycCamera()
            }
        }

        bottomBar.addView(btnCancel)
        bottomBar.addView(btnSnap)
        bottomBar.addView(btnFlip)

        val barParams = FrameLayout.LayoutParams(
            ViewGroup.LayoutParams.MATCH_PARENT,
            ViewGroup.LayoutParams.WRAP_CONTENT
        ).apply {
            gravity = Gravity.BOTTOM
        }
        overlay.addView(bottomBar, barParams)

        rootContainer.addView(
            overlay,
            FrameLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.MATCH_PARENT
            )
        )
        kycOverlayRoot = overlay
        bindKycCamera()
    }

    private fun bindKycCamera() {
        val pv = kycPreviewView ?: return
        val providerFuture = ProcessCameraProvider.getInstance(this)
        providerFuture.addListener({
            try {
                val provider = providerFuture.get()
                val preview = Preview.Builder().build().also {
                    it.setSurfaceProvider(pv.surfaceProvider)
                }
                val selector = CameraSelector.Builder().requireLensFacing(kycLensFacing).build()
                provider.unbindAll()
                provider.bindToLifecycle(this, selector, preview)
            } catch (e: Exception) {
                e.printStackTrace()
            }
        }, ContextCompat.getMainExecutor(this))
    }

    private fun captureAndVerifyKycFace(rootContainer: FrameLayout, overlay: FrameLayout) {
        val frameBmp = kycPreviewView?.bitmap
        if (frameBmp == null) {
            Toast.makeText(this, "कैमरा तैयार नहीं है", Toast.LENGTH_SHORT).show()
            return
        }

        val inputImage = InputImage.fromBitmap(frameBmp, 0)
        FaceMatcherEngine.detector.process(inputImage)
            .addOnSuccessListener { faces ->
                if (faces.isEmpty()) {
                    Toast.makeText(this, "❌ चेहरा साफ नहीं दिखा! सीधे देखें", Toast.LENGTH_SHORT).show()
                    return@addOnSuccessListener
                }
                val primaryFace = faces.maxByOrNull { it.boundingBox.width() * it.boundingBox.height() }!!
                val (isValid, qualityMsg) = FaceMatcherEngine.validateKycFaceQuality(primaryFace, frameBmp)
                if (!isValid) {
                    Toast.makeText(this, qualityMsg, Toast.LENGTH_LONG).show()
                    return@addOnSuccessListener
                }

                // 1. DP के लिए AI-Enhanced पूरा पोर्ट्रेट चेहरा
                val portraitDpBmp = FaceMatcherEngine.createAiEnhancedPortraitDp(frameBmp, primaryFace)
                // 2. 192-D बायोमेट्रिक डेटा के लिए अलाइन किया हुआ फेस
                val alignedBiometricFace = FaceMatcherEngine.alignAndCropFaceBitmap(frameBmp, primaryFace)

                showKycPhotoConfirmDialog(portraitDpBmp, alignedBiometricFace, rootContainer, overlay)
            }
            .addOnFailureListener {
                Toast.makeText(this, "❌ त्रुटि", Toast.LENGTH_SHORT).show()
            }
    }

    private fun showKycPhotoConfirmDialog(
        portraitDpBmp: Bitmap,
        alignedBiometricBmp: Bitmap,
        rootContainer: FrameLayout,
        overlay: FrameLayout
    ) {
        var workingDpBmp = portraitDpBmp
        var workingBioBmp = alignedBiometricBmp

        val imageView = ImageView(this).apply {
            setImageBitmap(workingDpBmp)
            adjustViewBounds = true
            maxHeight = 520
            setPadding(24, 24, 24, 12)
        }

        val btnRotate = Button(this).apply {
            text = "⤵️ फोटो घुमाएं (Rotate)"
            setOnClickListener {
                val matrix = Matrix().apply { postRotate(90f) }
                workingDpBmp = Bitmap.createBitmap(workingDpBmp, 0, 0, workingDpBmp.width, workingDpBmp.height, matrix, true)
                workingBioBmp = Bitmap.createBitmap(workingBioBmp, 0, 0, workingBioBmp.width, workingBioBmp.height, matrix, true)
                imageView.setImageBitmap(workingDpBmp)
            }
        }

        val box = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            gravity = Gravity.CENTER
            setPadding(30, 20, 30, 10)
            addView(imageView)
            addView(btnRotate)
        }

        AlertDialog.Builder(this)
            .setTitle("📸 KYC फोटो एवं 192-D बायोमेट्रिक पुष्टि")
            .setView(box)
            .setPositiveButton("✔ यह फोटो जोड़ें") { _, _ ->
                // बिना किसी जल्दबाज़ी के बैकग्राउंड में 192-D बायोमेट्रिक वेक्टर और DP तैयार करें
                Thread {
                    val embedding192D = FaceMatcherEngine.extractHighPrecisionKycEmbeddingList(workingBioBmp)

                    val scaledDp = Bitmap.createScaledBitmap(workingDpBmp, 260, 260, true)
                    val baos = ByteArrayOutputStream()
                    scaledDp.compress(Bitmap.CompressFormat.JPEG, 86, baos)
                    val base64Str = "data:image/jpeg;base64," + Base64.encodeToString(baos.toByteArray(), Base64.NO_WRAP)

                    val escapedPhoto = gson.toJson(base64Str)
                    val escapedEmb = gson.toJson(embedding192D)

                    runOnUiThread {
                        rootContainer.removeView(overlay)
                        kycOverlayRoot = null
                        webView.evaluateJavascript("onLiveKycPhotoApproved($escapedPhoto, $escapedEmb);", null)
                        Toast.makeText(
                            this@AdminActivity,
                            "🧬 DP + 192-D बायोमेट्रिक डेटा (${embedding192D.size} पॉइंट्स) तैयार!",
                            Toast.LENGTH_SHORT
                        ).show()
                    }
                }.start()
            }
            .setNegativeButton("दोबारा लें", null)
            .show()
    }

    inner class AdminWebBridge {

        @JavascriptInterface
        fun loadNativePayload(): String {
            val payload = AppStorage.loadData(this@AdminActivity)
            return gson.toJson(payload)
        }

        @JavascriptInterface
        fun saveNativePayload(jsonStr: String) {
            try {
                val parsed = gson.fromJson(jsonStr, BackupPayload::class.java)
                if (parsed != null) {
                    val existing = AppStorage.loadData(this@AdminActivity)
                    if (parsed.sharedApiKeys.isBlank() && existing.sharedApiKeys.isNotBlank()) {
                        parsed.sharedApiKeys = existing.sharedApiKeys
                    }
                    AppStorage.saveData(this@AdminActivity, parsed)
                    FaceMatcherEngine.clearCache()
                    FaceMatcherEngine.warmUpCache(parsed.students)
                }
            } catch (e: Exception) {
                e.printStackTrace()
            }
        }

        @JavascriptInterface
        fun openMultiApiKeyDialog() {
            runOnUiThread {
                val vm = RealisticVoiceManager(this@AdminActivity)

                val container = LinearLayout(this@AdminActivity).apply {
                    orientation = LinearLayout.VERTICAL
                    setPadding(45, 30, 45, 16)
                }

                val lblGemini = TextView(this@AdminActivity).apply {
                    text = "🔹 Google AI Studio (Gemini) Keys:"
                    textSize = 13.5f
                    setTypeface(null, Typeface.BOLD)
                    setTextColor(Color.parseColor("#1E3A8A"))
                    setPadding(0, 0, 0, 10)
                }

                val inputGemini = EditText(this@AdminActivity).apply {
                    setText(vm.getGeminiKeysString())
                    hint = "AQ.Ab8... या AIza... (कॉमा या नई लाइन से)"
                    textSize = 13.5f
                    minLines = 2
                    maxLines = 4
                    setPadding(28, 22, 28, 22)
                    background = GradientDrawable().apply {
                        setColor(Color.parseColor("#F8FAFC"))
                        cornerRadius = 18f
                        setStroke(2, Color.parseColor("#CBD5E1"))
                    }
                }

                val lblEleven = TextView(this@AdminActivity).apply {
                    text = "🔸 ElevenLabs API Keys:"
                    textSize = 13.5f
                    setTypeface(null, Typeface.BOLD)
                    setTextColor(Color.parseColor("#15803D"))
                    setPadding(0, 28, 0, 10)
                }

                val inputEleven = EditText(this@AdminActivity).apply {
                    setText(vm.getElevenLabsKeysString())
                    hint = "sk_... वाली Keys यहाँ डालें"
                    textSize = 13.5f
                    minLines = 2
                    maxLines = 4
                    setPadding(28, 22, 28, 22)
                    background = GradientDrawable().apply {
                        setColor(Color.parseColor("#F0FDF4"))
                        cornerRadius = 18f
                        setStroke(2, Color.parseColor("#86EFAC"))
                    }
                }

                container.addView(lblGemini)
                container.addView(inputGemini)
                container.addView(lblEleven)
                container.addView(inputEleven)

                AlertDialog.Builder(this@AdminActivity)
                    .setTitle("🎙️ AI Voice API Keys प्रबंधन")
                    .setView(container)
                    .setPositiveButton("✔ सेव करें") { _, _ ->
                        val gKeys = inputGemini.text.toString().trim()
                        val eKeys = inputEleven.text.toString().trim()
                        vm.saveSeparateApiKeys(gKeys, eKeys)

                        val combined = listOf(gKeys, eKeys).filter { it.isNotEmpty() }.joinToString("\n")
                        val currentPayload = AppStorage.loadData(this@AdminActivity)
                        currentPayload.sharedApiKeys = combined
                        AppStorage.saveData(this@AdminActivity, currentPayload)

                        Toast.makeText(this@AdminActivity, "✔ दोनों बॉक्स की Keys सुरक्षित सेव हो गईं", Toast.LENGTH_SHORT).show()
                    }
                    .setNegativeButton("रद्द करें", null)
                    .show()
            }
        }

        @JavascriptInterface
        fun closeAdminAndOpenScanner() {
            runOnUiThread {
                finish()
            }
        }

        @JavascriptInterface
        fun openLiveKycCamera() {
            runOnUiThread {
                val root = window.decorView.findViewById<ViewGroup>(android.R.id.content) as? FrameLayout
                if (root != null) {
                    showLiveKycCameraOverlay(root)
                }
            }
        }

        @JavascriptInterface
        fun openNativeRestorePicker() {
            runOnUiThread {
                pendingRestoreMode = "RESTORE"
                filePickerLauncher.launch("*/*")
            }
        }

        @JavascriptInterface
        fun openNativeMergePicker() {
            runOnUiThread {
                pendingRestoreMode = "MERGE"
                filePickerLauncher.launch("*/*")
            }
        }

        @JavascriptInterface
        fun exportScannerPackNative(targetClass: String) {
            runOnUiThread {
                val master = AppStorage.loadData(this@AdminActivity)
                val filteredStudents = if (targetClass == "सभी") {
                    master.students
                } else {
                    master.students.filter { it.className == targetClass }.toMutableList()
                }
                if (filteredStudents.isEmpty()) {
                    Toast.makeText(this@AdminActivity, "⚠️ कोई छात्र नहीं है", Toast.LENGTH_SHORT).show()
                    return@runOnUiThread
                }

                val packPayload = BackupPayload(
                    students = filteredStudents,
                    attendance = master.attendance,
                    holidays = master.holidays,
                    adminPin = master.adminPin,
                    isScannerOnlyPack = true,
                    sharedApiKeys = master.sharedApiKeys
                )
                val json = gson.toJson(packPayload)
                shareTextFileDirect("ScannerPack_${targetClass}_${AppStorage.getTodayShortKey().replace(".", "_")}.json", json, "application/json")
            }
        }

        @JavascriptInterface
        fun exportMasterBackupNative() {
            runOnUiThread {
                val master = AppStorage.loadData(this@AdminActivity)
                master.isScannerOnlyPack = false
                val json = gson.toJson(master)
                shareTextFileDirect("SMV_MasterBackup_${AppStorage.getTodayShortKey().replace(".", "_")}.json", json, "application/json")
            }
        }

        @JavascriptInterface
        fun saveMasterBackupToPhoneNative() {
            runOnUiThread {
                val master = AppStorage.loadData(this@AdminActivity)
                master.isScannerOnlyPack = false
                val json = gson.toJson(master)
                val fileName = "SMV_MasterBackup_${AppStorage.getTodayShortKey().replace(".", "_")}.json"
                saveJsonToDownloads(fileName, json)
            }
        }

        @JavascriptInterface
        fun shareFileContent(fileName: String, content: String, mimeType: String) {
            runOnUiThread {
                shareTextFileDirect(fileName, content, mimeType)
            }
        }

        @JavascriptInterface
        fun printHtmlDocument(htmlContent: String, jobName: String) {
            runOnUiThread {
                val pWeb = WebView(this@AdminActivity)
                pWeb.settings.javaScriptEnabled = false
                pWeb.webViewClient = object : WebViewClient() {
                    override fun onPageFinished(view: WebView?, url: String?) {
                        val printManager = getSystemService(Context.PRINT_SERVICE) as? PrintManager
                        val adapter = pWeb.createPrintDocumentAdapter(jobName)
                        printManager?.print(jobName, adapter, PrintAttributes.Builder().build())
                        printWebView = null
                    }
                }
                printWebView = pWeb
                pWeb.loadDataWithBaseURL(null, htmlContent, "text/HTML", "UTF-8", null)
            }
        }
    }

    private fun shareTextFileDirect(fileName: String, content: String, mimeType: String) {
        try {
            val outFile = File(cacheDir, fileName)
            FileOutputStream(outFile).use { fos ->
                fos.write(content.toByteArray(Charsets.UTF_8))
            }
            val uri = FileProvider.getUriForFile(
                this,
                "${applicationContext.packageName}.fileprovider",
                outFile
            )
            val intent = Intent(Intent.ACTION_SEND).apply {
                type = mimeType
                putExtra(Intent.EXTRA_STREAM, uri)
                clipData = ClipData.newRawUri(fileName, uri)
                addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
            }
            startActivity(Intent.createChooser(intent, "फाइल शेयर करें"))
        } catch (e: Exception) {
            Toast.makeText(this, "❌ शेयर त्रुटि", Toast.LENGTH_SHORT).show()
        }
    }

    private fun saveJsonToDownloads(fileName: String, jsonContent: String) {
        try {
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                val values = ContentValues().apply {
                    put(MediaStore.MediaColumns.DISPLAY_NAME, fileName)
                    put(MediaStore.MediaColumns.MIME_TYPE, "application/json")
                    put(MediaStore.MediaColumns.RELATIVE_PATH, Environment.DIRECTORY_DOWNLOADS)
                }
                val uri = contentResolver.insert(MediaStore.Downloads.EXTERNAL_CONTENT_URI, values)
                if (uri != null) {
                    contentResolver.openOutputStream(uri)?.use { os ->
                        os.write(jsonContent.toByteArray(Charsets.UTF_8))
                    }
                    Toast.makeText(this, "📥 Downloads में सेव हो गया: $fileName", Toast.LENGTH_LONG).show()
                }
            } else {
                val dir = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS)
                val outFile = File(dir, fileName)
                FileOutputStream(outFile).use { fos ->
                    fos.write(jsonContent.toByteArray(Charsets.UTF_8))
                }
                Toast.makeText(this, "📥 Downloads में सेव हो गया: $fileName", Toast.LENGTH_LONG).show()
            }
        } catch (e: Exception) {
            shareTextFileDirect(fileName, jsonContent, "application/json")
        }
    }
}
