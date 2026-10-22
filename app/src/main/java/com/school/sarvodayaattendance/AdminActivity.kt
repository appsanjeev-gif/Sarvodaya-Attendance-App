package com.school.sarvodayaattendance

import android.annotation.SuppressLint
import android.app.Activity
import android.content.Context
import android.content.Intent
import android.net.Uri
import android.os.Bundle
import android.print.PrintAttributes
import android.print.PrintManager
import android.webkit.JavascriptInterface
import android.webkit.ValueCallback
import android.webkit.WebChromeClient
import android.webkit.WebSettings
import android.webkit.WebView
import android.webkit.WebViewClient
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import com.google.gson.Gson
import com.google.gson.reflect.TypeToken
import com.school.sarvodayaattendance.data.AppStorage
import com.school.sarvodayaattendance.data.BackupPayload
import com.school.sarvodayaattendance.data.Student
import com.school.sarvodayaattendance.utils.FaceMatcherEngine
import java.io.BufferedReader
import java.io.InputStreamReader

class AdminActivity : AppCompatActivity() {

    private lateinit var webView: WebView
    private var filePathCallback: ValueCallback<Array<Uri>>? = null
    private val gson = Gson()

    // Native Restore Picker for 200KB+ HTML/JSON backup files
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
        webView = WebView(this)
        setContentView(webView)

        val ws: WebSettings = webView.settings
        ws.javaScriptEnabled = true
        ws.domStorageEnabled = true
        ws.allowFileAccess = true
        ws.allowContentAccess = true
        ws.allowFileAccessFromFileURLs = true
        ws.allowUniversalAccessFromFileURLs = true

        webView.addJavascriptInterface(NativeAdminBridge(), "AndroidBridge")
        webView.webViewClient = WebViewClient()
        webView.webChromeClient = object : WebChromeClient() {
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
                        putExtra(Intent.EXTRA_ALLOW_MULTIPLE, true)
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
                try {
                    val sendIntent = Intent(Intent.ACTION_SEND).apply {
                        type = "text/plain"
                        putExtra(Intent.EXTRA_SUBJECT, fileName)
                        putExtra(Intent.EXTRA_TEXT, content)
                    }
                    startActivity(Intent.createChooser(sendIntent, fileName))
                } catch (e: Exception) {
                    Toast.makeText(this@AdminActivity, "शेयर त्रुटि!", Toast.LENGTH_SHORT).show()
                }
            }
        }
    }

    @Deprecated("Deprecated in Java")
    override fun onBackPressed() {
        finish()
    }
}
