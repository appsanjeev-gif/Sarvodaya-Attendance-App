package com.school.sarvodayaattendance

import android.annotation.SuppressLint
import android.app.Activity
import android.content.Context
import android.content.Intent
import android.graphics.Bitmap
import android.net.Uri
import android.os.Bundle
import android.os.StrictMode
import android.print.PrintAttributes
import android.print.PrintManager
import android.util.Base64
import android.view.Gravity
import android.view.ViewGroup
import android.webkit.JavascriptInterface
import android.webkit.PermissionRequest
import android.webkit.ValueCallback
import android.webkit.WebChromeClient
import android.webkit.WebSettings
import android.webkit.WebView
import android.webkit.WebViewClient
import android.widget.Button
import android.widget.FrameLayout
import android.widget.LinearLayout
import android.widget.ProgressBar
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import com.google.gson.Gson
import com.google.mlkit.vision.common.InputImage
import com.school.sarvodayaattendance.data.AppStorage
import com.school.sarvodayaattendance.utils.FaceMatcherEngine
import java.io.BufferedReader
import java.io.ByteArrayOutputStream
import java.io.File
import java.io.FileOutputStream
import java.io.InputStreamReader
import java.io.OutputStreamWriter

class AdminActivity : AppCompatActivity() {

    private lateinit var webView: WebView
    private var filePathCallback: ValueCallback<Array<Uri>>? = null
    private val gson = Gson()

    private var pendingSaveContent: String = ""
    private var pendingSaveFileName: String = "Backup.txt"

    // 1. Native Restore Picker
    private val nativeRestoreLauncher = registerForActivityResult(
        ActivityResultContracts.GetContent()
    ) { uri: Uri? ->
        if (uri != null) {
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
                    FaceMatcherEngine.clearCache()
                    val count = AppStorage.loadData(this).students.size
                    Toast.makeText(this, "✔ डेटा रिस्टोर हो गया! (कुल छात्र: $count)", Toast.LENGTH_LONG).show()
                    webView.reload()
                } else {
                    Toast.makeText(this, "❌ अमान्य बैकअप फ़ाइल!", Toast.LENGTH_SHORT).show()
                }
            } catch (e: Exception) {
                Toast.makeText(this, "❌ फ़ाइल पढ़ने में त्रुटि!", Toast.LENGTH_SHORT).show()
            }
        }
    }

    // 2. Native Merge Daily Attendance Picker (दूसरे मोबाइल की हाज़िरी मर्ज करने के लिए)
    private val nativeMergeAttendanceLauncher = registerForActivityResult(
        ActivityResultContracts.GetContent()
    ) { uri: Uri? ->
        if (uri != null) {
            try {
                val sb = StringBuilder()
                contentResolver.openInputStream(uri)?.use { stream ->
                    BufferedReader(InputStreamReader(stream)).use { reader ->
                        var line: String?
                        while (reader.readLine().also { line = it } != null) {
                            sb.append(line)
                        }
                    }
                }
                val rawJson = sb.toString().trim()
                val escaped = gson.toJson(rawJson)
                webView.evaluateJavascript("onNativeAttendanceMergeContent($escaped)", null)
            } catch (e: Exception) {
                Toast.makeText(this, "❌ हाज़िरी फ़ाइल पढ़ने में त्रुटि!", Toast.LENGTH_SHORT).show()
            }
        }
    }

    // 3. Native Save File Picker (फोन के Downloads में सीधे सेव करने के लिए)
    private val createDocumentLauncher = registerForActivityResult(
        ActivityResultContracts.StartActivityForResult()
    ) { result ->
        if (result.resultCode == Activity.RESULT_OK && result.data?.data != null) {
            val uri = result.data!!.data!!
            try {
                contentResolver.openOutputStream(uri)?.use { out ->
                    OutputStreamWriter(out, Charsets.UTF_8).use { writer ->
                        writer.write(pendingSaveContent)
                    }
                }
                Toast.makeText(this, "✔ फ़ाइल फोन में सफलतापूर्वक सेव हो गई!", Toast.LENGTH_LONG).show()
            } catch (e: Exception) {
                Toast.makeText(this, "❌ सेव करने में त्रुटि!", Toast.LENGTH_SHORT).show()
            }
        }
    }

    private val fileChooserLauncher = registerForActivityResult(
        ActivityResultContracts.StartActivityForResult()
    ) { result ->
        if (filePathCallback == null) return@registerForActivityResult
        var results: Array<Uri>? = null
        if (result.resultCode == Activity.RESULT_OK && result.data != null) {
            val dataString = result.data?.dataString
            val clipData = result.data?.clipData
            if (clipData != null) {
                results = Array(clipData.itemCount) { i -> clipData.getItemAt(i).uri }
            } else if (dataString != null) {
                results = arrayOf(Uri.parse(dataString))
            }
        }
        filePathCallback?.onReceiveValue(results)
        filePathCallback = null
    }

    @SuppressLint("SetJavaScriptEnabled")
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        StrictMode.setVmPolicy(StrictMode.VmPolicy.Builder().build())

        webView = WebView(this)
        setContentView(webView)

        val ws: WebSettings = webView.settings
        ws.javaScriptEnabled = true
        ws.domStorageEnabled = true
        ws.allowFileAccess = true
        ws.allowContentAccess = true
        ws.allowFileAccessFromFileURLs = true
        ws.allowUniversalAccessFromFileURLs = true
        ws.mediaPlaybackRequiresUserGesture = false

        webView.addJavascriptInterface(NativeAdminBridge(), "AndroidBridge")
        webView.webViewClient = WebViewClient()
        webView.webChromeClient = object : WebChromeClient() {
            override fun onPermissionRequest(request: PermissionRequest?) {
                runOnUiThread {
                    request?.grant(request.resources)
                }
            }

            override fun onShowFileChooser(
                webView: WebView?,
                filePathCallback: ValueCallback<Array<Uri>>?,
                fileChooserParams: FileChooserParams?
            ): Boolean {
                this@AdminActivity.filePathCallback?.onReceiveValue(null)
                this@AdminActivity.filePathCallback = filePathCallback
                return try {
                    val intent = Intent(Intent.ACTION_GET_CONTENT).apply {
                        addCategory(Intent.CATEGORY_OPENABLE)
                        type = "*/*"
                    }
                    fileChooserLauncher.launch(Intent.createChooser(intent, "फाइल चुनें"))
                    true
                } catch (e: Exception) {
                    this@AdminActivity.filePathCallback = null
                    false
                }
            }
        }

        webView.loadUrl("file:///android_asset/admin.html")
    }

    private fun showNativeLiveKycDialog() {
        val dialogLayout = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            gravity = Gravity.CENTER_HORIZONTAL
            setPadding(36, 36, 36, 36)
            setBackgroundColor(0xFF0F172A.toInt())
        }

        val titleTv = TextView(this).apply {
            text = "📸 लाइव बायोमेट्रिक KYC वेरिफिकेशन"
            textSize = 17f
            setTextColor(0xFFFFFFFF.toInt())
            gravity = Gravity.CENTER
        }

        val subTv = TextView(this).apply {
            text = "चेहरे को गोले के बीच में सीधा रखें..."
            textSize = 13.5f
            setTextColor(0xFF38BDF8.toInt())
            gravity = Gravity.CENTER
            setPadding(0, 10, 0, 20)
        }

        val circleContainer = FrameLayout(this).apply {
            layoutParams = LinearLayout.LayoutParams(620, 620).apply {
                gravity = Gravity.CENTER
            }
        }

        val kycPreview = PreviewView(this).apply {
            layoutParams = FrameLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.MATCH_PARENT
            )
            clipToOutline = true
        }

        val progressRing = ProgressBar(this, null, android.R.attr.progressBarStyleHorizontal).apply {
            layoutParams = FrameLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                22,
                Gravity.BOTTOM
            )
            max = 100
            progress = 0
        }

        circleContainer.addView(kycPreview)
        circleContainer.addView(progressRing)

        val btnRow = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER
            setPadding(0, 28, 0, 0)
        }

        val switchCamBtn = Button(this).apply {
            text = "🔄 कैमरा बदलें"
            setBackgroundColor(0xFF334155.toInt())
            setTextColor(0xFFFFFFFF.toInt())
        }

        val cancelBtn = Button(this).apply {
            text = "✕ बंद करें"
            setBackgroundColor(0xFFDC2626.toInt())
            setTextColor(0xFFFFFFFF.toInt())
        }

        btnRow.addView(switchCamBtn)
        btnRow.addView(cancelBtn)

        dialogLayout.addView(titleTv)
        dialogLayout.addView(subTv)
        dialogLayout.addView(circleContainer)
        dialogLayout.addView(btnRow)

        val kycDialog = AlertDialog.Builder(this)
            .setView(dialogLayout)
            .setCancelable(false)
            .create()

        var kycLens = CameraSelector.LENS_FACING_BACK
        var isDialogActive = true
        var approvalProgress = 0
        var isChecking = false

        fun bindKycCamera() {
            val fut = ProcessCameraProvider.getInstance(this)
            fut.addListener({
                try {
                    val provider = fut.get()
                    val preview = Preview.Builder().build().also {
                        it.setSurfaceProvider(kycPreview.surfaceProvider)
                    }
                    val sel = CameraSelector.Builder().requireLensFacing(kycLens).build()
                    provider.unbindAll()
                    provider.bindToLifecycle(this, sel, preview)
                } catch (e: Exception) {
                    e.printStackTrace()
                }
            }, ContextCompat.getMainExecutor(this))
        }

        switchCamBtn.setOnClickListener {
            kycLens = if (kycLens == CameraSelector.LENS_FACING_FRONT) {
                CameraSelector.LENS_FACING_BACK
            } else {
                CameraSelector.LENS_FACING_FRONT
            }
            approvalProgress = 0
            progressRing.progress = 0
            bindKycCamera()
        }

        cancelBtn.setOnClickListener {
            isDialogActive = false
            kycDialog.dismiss()
        }

        val checkRunnable = object : Runnable {
            override fun run() {
                if (!isDialogActive) return
                val bmp = kycPreview.bitmap
                if (bmp != null && !isChecking) {
                    isChecking = true
                    val img = InputImage.fromBitmap(bmp, 0)
                    FaceMatcherEngine.detector.process(img)
                        .addOnSuccessListener { faces ->
                            if (!isDialogActive) return@addOnSuccessListener
                            if (faces.size == 1) {
                                val f = faces[0]
                                val yaw = kotlin.math.abs(f.headEulerAngleY)
                                val tilt = kotlin.math.abs(f.headEulerAngleZ)
                                val wRatio = f.boundingBox.width().toFloat() / bmp.width.toFloat()

                                if (yaw <= 16f && tilt <= 16f && wRatio >= 0.25f) {
                                    approvalProgress += 35
                                    progressRing.progress = approvalProgress.coerceAtMost(100)
                                    subTv.text = "🟢 बहुत बढ़िया! स्थिर रहें (${progressRing.progress}%)"
                                    subTv.setTextColor(0xFF4ADE80.toInt())

                                    if (approvalProgress >= 100) {
                                        isDialogActive = false
                                        val cropped = FaceMatcherEngine.cropFaceBitmap(bmp, f.boundingBox)
                                        val scaled = Bitmap.createScaledBitmap(cropped, 160, 160, true)
                                        val baos = ByteArrayOutputStream()
                                        scaled.compress(Bitmap.CompressFormat.JPEG, 85, baos)
                                        val b64 = "data:image/jpeg;base64," + Base64.encodeToString(baos.toByteArray(), Base64.NO_WRAP)

                                        kycDialog.dismiss()
                                        webView.evaluateJavascript("onLiveKycPhotoApproved('$b64')", null)
                                        Toast.makeText(this@AdminActivity, "✔ बायोमेट्रिक फोटो अप्रूव हो गई!", Toast.LENGTH_SHORT).show()
                                        return@addOnSuccessListener
                                    }
                                } else {
                                    approvalProgress = 0
                                    progressRing.progress = 0
                                    subTv.text = "⚠️ चेहरा गोले के बीच में बिल्कुल सीधा रखें!"
                                    subTv.setTextColor(0xFFFACC15.toInt())
                                }
                            } else {
                                approvalProgress = 0
                                progressRing.progress = 0
                                subTv.text = "🔍 कैमरे के सामने केवल छात्र का चेहरा लाएं..."
                                subTv.setTextColor(0xFFF87171.toInt())
                            }
                            isChecking = false
                        }
                        .addOnFailureListener {
                            isChecking = false
                        }
                }
                if (isDialogActive) {
                    kycPreview.postDelayed(this, 380)
                }
            }
        }

        kycDialog.show()
        bindKycCamera()
        kycPreview.postDelayed(checkRunnable, 600)
    }

    inner class NativeAdminBridge {

        @JavascriptInterface
        fun loadNativePayload(): String {
            val payload = AppStorage.loadData(this@AdminActivity)
            return gson.toJson(payload)
        }

        @JavascriptInterface
        fun openNativeRestorePicker() {
            runOnUiThread {
                nativeRestoreLauncher.launch("*/*")
            }
        }

        @JavascriptInterface
        fun openNativeMergePicker() {
            runOnUiThread {
                nativeMergeAttendanceLauncher.launch("*/*")
            }
        }

        @JavascriptInterface
        fun openLiveKycCamera() {
            runOnUiThread {
                showNativeLiveKycDialog()
            }
        }

        @JavascriptInterface
        fun saveNativePayload(jsonStr: String) {
            try {
                val ok = AppStorage.importFromHtmlOrJson(this@AdminActivity, jsonStr)
                if (ok) {
                    FaceMatcherEngine.clearCache()
                }
            } catch (e: Exception) {
                e.printStackTrace()
            }
        }

        // ✅ मास्टर बैकअप को WhatsApp-Compatible (.txt) फॉर्मेट में सेव या शेयर करें
        @JavascriptInterface
        fun exportMasterBackupNative() {
            runOnUiThread {
                try {
                    val payload = AppStorage.loadData(this@AdminActivity)
                    payload.isScannerOnlyPack = false
                    val jsonStr = gson.toJson(payload)
                    val dateClean = AppStorage.getTodayShortKey().replace(".", "_")
                    val fileName = "Sarvodaya_Master_Backup_$dateClean.txt"
                    showSaveOrShareDialog("💾 मास्टर बैकअप तैयार है", fileName, jsonStr, "text/plain")
                } catch (e: Exception) {
                    Toast.makeText(this@AdminActivity, "बैकअप त्रुटि!", Toast.LENGTH_SHORT).show()
                }
            }
        }

        // ✅ स्कैनर पैक को भी WhatsApp-Compatible (.txt) फॉर्मेट में सेव या शेयर करें (ताकि WhatsApp कभी ब्लॉक न करे)
        @JavascriptInterface
        fun exportScannerPackNative(targetClass: String) {
            runOnUiThread {
                try {
                    val fullPayload = AppStorage.loadData(this@AdminActivity)
                    val filteredStudents = if (targetClass == "सभी") {
                        fullPayload.students
                    } else {
                        fullPayload.students.filter { it.className == targetClass }.toMutableList()
                    }
                    if (filteredStudents.isEmpty()) {
                        Toast.makeText(this@AdminActivity, "⚠️ इस कक्षा में कोई छात्र नहीं है!", Toast.LENGTH_SHORT).show()
                        return@runOnUiThread
                    }
                    val packObj = mapOf(
                        "isScannerOnlyPack" to true,
                        "students" to filteredStudents,
                        "attendance" to emptyMap<String, Any>(),
                        "holidays" to fullPayload.holidays,
                        "adminPin" to fullPayload.adminPin
                    )
                    val jsonStr = gson.toJson(packObj)
                    val cleanCls = when (targetClass) {
                        "सभी" -> "All_Classes"
                        "6वीं" -> "Class_6"
                        "7वीं" -> "Class_7"
                        "8वीं" -> "Class_8"
                        else -> "Class_Pack"
                    }
                    val dateClean = AppStorage.getTodayShortKey().replace(".", "_")
                    val fileName = "Scanner_Pack_${cleanCls}_$dateClean.txt"
                    showSaveOrShareDialog("📤 स्कैनर पैक तैयार है (${filteredStudents.size} छात्र)", fileName, jsonStr, "text/plain")
                } catch (e: Exception) {
                    Toast.makeText(this@AdminActivity, "शेयर त्रुटि!", Toast.LENGTH_SHORT).show()
                }
            }
        }

        @JavascriptInterface
        fun closeAdminAndOpenScanner() {
            runOnUiThread {
                finish()
            }
        }

        @JavascriptInterface
        fun printHtmlDocument(htmlContent: String, jobName: String) {
            runOnUiThread {
                try {
                    val printWebView = WebView(this@AdminActivity)
                    printWebView.webViewClient = object : WebViewClient() {
                        override fun onPageFinished(view: WebView?, url: String?) {
                            val printManager = getSystemService(Context.PRINT_SERVICE) as PrintManager
                            val adapter = printWebView.createPrintDocumentAdapter(jobName)
                            printManager.print(jobName, adapter, PrintAttributes.Builder().build())
                        }
                    }
                    printWebView.loadDataWithBaseURL(null, htmlContent, "text/html", "UTF-8", null)
                } catch (e: Exception) {
                    Toast.makeText(this@AdminActivity, "प्रिंट त्रुटि!", Toast.LENGTH_SHORT).show()
                }
            }
        }

        @JavascriptInterface
        fun shareFileContent(fileName: String, content: String, mimeType: String) {
            runOnUiThread {
                val safeMime = if (mimeType.contains("json")) "text/plain" else mimeType
                val safeName = if (fileName.endsWith(".json")) fileName.replace(".json", ".txt") else fileName
                shareLargeFileDirectly(safeName, content, safeMime)
            }
        }
    }

    private fun showSaveOrShareDialog(title: String, fileName: String, content: String, mimeType: String) {
        AlertDialog.Builder(this)
            .setTitle(title)
            .setMessage("फ़ाइल: $fileName\nआप इसे कैसे भेजना या सेव करना चाहते हैं?")
            .setPositiveButton("📤 WhatsApp / शेयर करें") { _, _ ->
                shareLargeFileDirectly(fileName, content, mimeType)
            }
            .setNeutralButton("📥 फोन में सेव करें") { _, _ ->
                pendingSaveFileName = fileName
                pendingSaveContent = content
                val intent = Intent(Intent.ACTION_CREATE_DOCUMENT).apply {
                    addCategory(Intent.CATEGORY_OPENABLE)
                    type = mimeType
                    putExtra(Intent.EXTRA_TITLE, fileName)
                }
                createDocumentLauncher.launch(intent)
            }
            .setNegativeButton("रद्द", null)
            .show()
    }

    private fun shareLargeFileDirectly(fileName: String, content: String, mimeType: String) {
        try {
            val outFile = File(externalCacheDir ?: cacheDir, fileName)
            FileOutputStream(outFile).use { fos ->
                fos.write(content.toByteArray(Charsets.UTF_8))
            }
            val fileUri = Uri.fromFile(outFile)
            val sendIntent = Intent(Intent.ACTION_SEND).apply {
                type = mimeType
                putExtra(Intent.EXTRA_STREAM, fileUri)
                addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
            }
            startActivity(Intent.createChooser(sendIntent, fileName))
        } catch (e: Exception) {
            e.printStackTrace()
            Toast.makeText(this, "❌ फ़ाइल शेयर करने में त्रुटि!", Toast.LENGTH_SHORT).show()
        }
    }

    @Deprecated("Deprecated in Java")
    override fun onBackPressed() {
        finish()
    }
}
