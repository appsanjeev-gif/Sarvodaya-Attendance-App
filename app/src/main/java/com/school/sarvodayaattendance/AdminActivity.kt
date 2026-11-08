package com.school.sarvodayaattendance

import android.Manifest
import android.animation.ValueAnimator
import android.annotation.SuppressLint
import android.content.ClipData
import android.content.ContentValues
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Matrix
import android.graphics.Paint
import android.graphics.Path
import android.graphics.RectF
import android.graphics.Typeface
import android.graphics.drawable.GradientDrawable
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.os.Environment
import android.os.Handler
import android.os.Looper
import android.print.PrintAttributes
import android.print.PrintManager
import android.provider.MediaStore
import android.util.Base64
import android.view.Gravity
import android.view.View
import android.view.ViewGroup
import android.view.animation.DecelerateInterpolator
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
import com.school.sarvodayaattendance.utils.AiDiagnosticLogger
import com.school.sarvodayaattendance.utils.FaceMatcherEngine
import com.school.sarvodayaattendance.utils.RealisticVoiceManager
import java.io.BufferedReader
import java.io.ByteArrayOutputStream
import java.io.File
import java.io.FileOutputStream
import java.io.InputStreamReader
import kotlin.math.min

class AdminActivity : AppCompatActivity() {

    private lateinit var webView: WebView
    private var printWebView: WebView? = null
    private val gson = Gson()
    private val kycHandler = Handler(Looper.getMainLooper())

    private var kycOverlayRoot: FrameLayout? = null
    private var kycPreviewView: PreviewView? = null
    private var kycRingView: KycBiometricRingView? = null
    private var kycLensFacing = CameraSelector.LENS_FACING_BACK
    private var pendingRestoreMode = "RESTORE"

    @Volatile
    private var isKycMonitoringActive = false
    @Volatile
    private var isKycFrameBusy = false
    @Volatile
    private var isKycCaptureLocked = false
    private var consecutivePerfectKycFrames = 0

    // बिना किसी टेक्स्ट/नोटिफिकेशन के लाइव बायोमेट्रिक सर्कल एनीमेशन और 100% पूरा होने पर ऑटो-कैप्चर लूप
    private val kycMonitorRunnable = object : Runnable {
        override fun run() {
            if (!isKycMonitoringActive || kycOverlayRoot == null) return
            evaluateLiveKycFrameForCircle()
            if (isKycMonitoringActive) {
                kycHandler.postDelayed(this, 140L)
            }
        }
    }

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
                    val dim = FaceMatcherEngine.getEmbeddingDimension()
                    for (s in loaded.students) {
                        s.embeddings.removeAll { it.size != dim }
                    }
                    AppStorage.saveData(this, loaded)
                    FaceMatcherEngine.clearCache()
                    FaceMatcherEngine.warmUpCache(loaded.students, this)
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

    // कस्टम सर्कल रिंग एनीमेशन व्यू (बिना किसी टेक्स्ट या नोटिफिकेशन के केवल सर्कल भरना और पूरा होने पर हरा ग्लो)
    private inner class KycBiometricRingView(context: Context) : View(context) {
        private var animatedProgress = 0f
        private var isCompletedState = false
        private var fillPulseAlpha = 0
        private var progressAnimator: ValueAnimator? = null

        private val overlayPath = Path()
        private val arcRect = RectF()

        private val dimPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = Color.parseColor("#B80F172A")
            style = Paint.Style.FILL
        }

        private val trackPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = Color.parseColor("#38475569")
            style = Paint.Style.STROKE
            strokeWidth = 18f
        }

        private val guideDashPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = Color.parseColor("#5594A3B8")
            style = Paint.Style.STROKE
            strokeWidth = 3f
        }

        private val progressPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            style = Paint.Style.STROKE
            strokeWidth = 20f
            strokeCap = Paint.Cap.ROUND
        }

        private val glowFillPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            style = Paint.Style.FILL
            color = Color.parseColor("#22C55E")
        }

        fun updateProgressSmooth(targetFraction: Float, isComplete: Boolean) {
            val clamped = targetFraction.coerceIn(0f, 1f)
            isCompletedState = isComplete

            progressAnimator?.cancel()
            progressAnimator = ValueAnimator.ofFloat(animatedProgress, clamped).apply {
                duration = if (isComplete) 180L else 130L
                interpolator = DecelerateInterpolator()
                addUpdateListener { anim ->
                    animatedProgress = anim.animatedValue as Float
                    fillPulseAlpha = if (isCompletedState && animatedProgress >= 0.98f) 52 else 0
                    invalidate()
                }
                start()
            }
        }

        override fun onDraw(canvas: Canvas) {
            super.onDraw(canvas)
            val w = width.toFloat()
            val h = height.toFloat()
            if (w <= 0f || h <= 0f) return

            val cx = w * 0.5f
            val cy = h * 0.44f
            val radius = min(w, h) * 0.36f

            // 1. सर्कल के बाहर हल्का डार्क विग्नेट
            overlayPath.reset()
            overlayPath.addRect(0f, 0f, w, h, Path.Direction.CW)
            overlayPath.addCircle(cx, cy, radius, Path.Direction.CCW)
            overlayPath.fillType = Path.FillType.EVEN_ODD
            canvas.drawPath(overlayPath, dimPaint)

            // 2. जब सर्कल 100% पूरा भर जाए, तो अंदर हल्का हरा ग्लो फिल
            if (fillPulseAlpha > 0) {
                glowFillPaint.alpha = fillPulseAlpha
                canvas.drawCircle(cx, cy, radius - 6f, glowFillPaint)
            }

            // 3. कंधे और सिर की सीध के लिए सूक्ष्म गाइड आर्क (बिना किसी टेक्स्ट के)
            canvas.drawCircle(cx, cy, radius - 14f, guideDashPaint)

            // 4. सर्कल का बैकग्राउंड ट्रैक
            canvas.drawCircle(cx, cy, radius, trackPaint)

            // 5. लाइव प्रोग्रेस सर्कल आर्क (अधूरे पर अंबर/सियान, 100% पूरे होने पर चमकदार हरा)
            if (animatedProgress > 0.01f) {
                progressPaint.color = when {
                    isCompletedState || animatedProgress >= 0.99f -> Color.parseColor("#22C55E")
                    animatedProgress >= 0.65f -> Color.parseColor("#38BDF8")
                    else -> Color.parseColor("#F59E0B")
                }
                arcRect.set(cx - radius, cy - radius, cx + radius, cy + radius)
                val sweepAngle = (animatedProgress * 360f).coerceIn(0f, 360f)
                canvas.drawArc(arcRect, -90f, sweepAngle, false, progressPaint)
            }
        }
    }

    private fun stopKycMonitoring() {
        isKycMonitoringActive = false
        isKycFrameBusy = false
        consecutivePerfectKycFrames = 0
        kycHandler.removeCallbacks(kycMonitorRunnable)
    }

    private fun showLiveKycCameraOverlay(rootContainer: FrameLayout) {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this, arrayOf(Manifest.permission.CAMERA), 202)
            return
        }

        stopKycMonitoring()
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

        // सर्कल एनीमेशन व्यू जोड़ें
        val ringView = KycBiometricRingView(this)
        kycRingView = ringView
        overlay.addView(
            ringView,
            FrameLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.MATCH_PARENT
            )
        )

        val bottomBar = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER
            setPadding(36, 28, 36, 48)
            setBackgroundColor(0x88000000.toInt())
        }

        val btnCancel = Button(this).apply {
            text = "✕"
            textSize = 18f
            setTextColor(0xFFFFFFFF.toInt())
            background = GradientDrawable().apply {
                setColor(0xFFDC2626.toInt())
                cornerRadius = 28f
            }
            val lp = LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f).apply {
                setMargins(0, 0, 20, 0)
            }
            layoutParams = lp
            setOnClickListener {
                stopKycMonitoring()
                rootContainer.removeView(overlay)
                kycOverlayRoot = null
            }
        }

        val btnFlip = Button(this).apply {
            text = "🔄"
            textSize = 18f
            setTextColor(0xFFFFFFFF.toInt())
            background = GradientDrawable().apply {
                setColor(0xFF334155.toInt())
                cornerRadius = 28f
            }
            val lp = LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f).apply {
                setMargins(20, 0, 0, 0)
            }
            layoutParams = lp
            setOnClickListener {
                consecutivePerfectKycFrames = 0
                kycRingView?.updateProgressSmooth(0f, false)
                kycLensFacing = if (kycLensFacing == CameraSelector.LENS_FACING_FRONT) {
                    CameraSelector.LENS_FACING_BACK
                } else {
                    CameraSelector.LENS_FACING_FRONT
                }
                bindKycCamera()
            }
        }

        bottomBar.addView(btnCancel)
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
        isKycCaptureLocked = false
        consecutivePerfectKycFrames = 0
        bindKycCamera()

        isKycMonitoringActive = true
        kycHandler.postDelayed(kycMonitorRunnable, 280L)
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

    // हर फ्रेम पर 5° कोण, दोनों कान, सीधी नाक और खुली आँखों की जाँच कर सर्कल भरना और 100% होते ही ऑटो-कैप्चर करना
    private fun evaluateLiveKycFrameForCircle() {
        if (isKycFrameBusy || isKycCaptureLocked) return
        val overlay = kycOverlayRoot ?: return
        val rootContainer = window.decorView.findViewById<ViewGroup>(android.R.id.content) as? FrameLayout ?: return
        val rawFrameBmp = kycPreviewView?.bitmap ?: return

        isKycFrameBusy = true
        val detectBmp = FaceMatcherEngine.downscaleFrameForFastDetection(rawFrameBmp, 480)
        val inputImage = InputImage.fromBitmap(detectBmp, 0)

        FaceMatcherEngine.detector.process(inputImage)
            .addOnSuccessListener { faces ->
                if (!isKycMonitoringActive || isKycCaptureLocked) {
                    isKycFrameBusy = false
                    return@addOnSuccessListener
                }

                if (faces.isEmpty()) {
                    consecutivePerfectKycFrames = 0
                    kycRingView?.updateProgressSmooth(0f, false)
                    isKycFrameBusy = false
                    return@addOnSuccessListener
                }

                val primaryFace = faces.maxByOrNull { it.boundingBox.width() * it.boundingBox.height() }!!
                val status = FaceMatcherEngine.evaluateStrictKycProgress(
                    primaryFace,
                    detectBmp.width,
                    detectBmp.height
                )

                if (status.isPerfectFrame) {
                    consecutivePerfectKycFrames++
                    // 3 लगातार परफेक्ट फ्रेम (~0.42 सेकंड स्थिर रहने पर) सर्कल 88% -> 94% -> 100% पूरा भरकर ऑटो-कैप्चर करेगा
                    val stepFraction = when (consecutivePerfectKycFrames) {
                        1 -> 0.90f
                        2 -> 0.96f
                        else -> 1.0f
                    }
                    val isDone = consecutivePerfectKycFrames >= 3
                    kycRingView?.updateProgressSmooth(stepFraction, isDone)

                    if (isDone) {
                        isKycCaptureLocked = true
                        stopKycMonitoring()
                        kycHandler.postDelayed({
                            processApprovedKycCapture(rawFrameBmp, detectBmp, primaryFace, rootContainer, overlay)
                        }, 190L)
                    }
                } else {
                    consecutivePerfectKycFrames = 0
                    kycRingView?.updateProgressSmooth(status.progressFraction, false)
                }

                isKycFrameBusy = false
            }
            .addOnFailureListener {
                isKycFrameBusy = false
            }
    }

    private fun processApprovedKycCapture(
        rawFrameBmp: Bitmap,
        detectBmp: Bitmap,
        primaryFace: Face,
        rootContainer: FrameLayout,
        overlay: FrameLayout
    ) {
        // 1. कंधों सहित पासपोर्ट/आईडी स्टाइल 1:1 स्क्वायर क्रॉप (400x400)
        val rawSquareBmp = FaceMatcherEngine.cropTrueSquarePortraitFromFrames(rawFrameBmp, detectBmp, primaryFace)
        // 2. अपग्रेडेड ऑन-डिवाइस स्टूडियो ब्यूटी फिल्टर
        val previewDpBmp = FaceMatcherEngine.applyOnDeviceStudioPortraitFilter(rawSquareBmp)
        // 3. 480p और फुल-एचडी दोनों से सटीक 112x112 बायोमेट्रिक अलाइनमेंट
        val alignedBiometricFace = FaceMatcherEngine.alignAndCropFaceBitmap(detectBmp, primaryFace)
        val hdAlignedBiometricFace = FaceMatcherEngine.alignAndCropFaceBitmapFromFrames(rawFrameBmp, detectBmp, primaryFace)

        showKycPhotoConfirmDialog(
            rawSquareBmp,
            previewDpBmp,
            alignedBiometricFace,
            hdAlignedBiometricFace,
            rootContainer,
            overlay
        )
    }

    private fun showKycPhotoConfirmDialog(
        rawSquareBmp: Bitmap,
        previewDpBmp: Bitmap,
        alignedBiometricBmp: Bitmap,
        hdAlignedBiometricBmp: Bitmap,
        rootContainer: FrameLayout,
        overlay: FrameLayout
    ) {
        var workingRawSquare = rawSquareBmp
        var workingPreviewDp = previewDpBmp
        var workingBioBmp = alignedBiometricBmp
        var workingHdBioBmp = hdAlignedBiometricBmp

        val imageView = ImageView(this).apply {
            setImageBitmap(workingPreviewDp)
            adjustViewBounds = true
            maxHeight = 540
            setPadding(24, 24, 24, 12)
        }

        val btnRotate = Button(this).apply {
            text = "⤵️ फोटो घुमाएं (Rotate)"
            setOnClickListener {
                val matrix = Matrix().apply { postRotate(90f) }
                workingRawSquare = Bitmap.createBitmap(workingRawSquare, 0, 0, workingRawSquare.width, workingRawSquare.height, matrix, true)
                workingPreviewDp = Bitmap.createBitmap(workingPreviewDp, 0, 0, workingPreviewDp.width, workingPreviewDp.height, matrix, true)
                workingBioBmp = Bitmap.createBitmap(workingBioBmp, 0, 0, workingBioBmp.width, workingBioBmp.height, matrix, true)
                workingHdBioBmp = Bitmap.createBitmap(workingHdBioBmp, 0, 0, workingHdBioBmp.width, workingHdBioBmp.height, matrix, true)
                imageView.setImageBitmap(workingPreviewDp)
            }
        }

        val box = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            gravity = Gravity.CENTER
            setPadding(30, 20, 30, 10)
            addView(imageView)
            addView(btnRotate)
        }

        val dialog = AlertDialog.Builder(this)
            .setTitle("📸 1:1 स्टूडियो DP एवं 192-D पुष्टि")
            .setView(box)
            .setCancelable(false)
            .setPositiveButton("✔ यह फोटो जोड़ें") { _, _ ->
                Thread {
                    val embedding192D = FaceMatcherEngine.extractHighPrecisionKycEmbeddingList(
                        workingBioBmp,
                        workingHdBioBmp
                    )

                    val (finalDpBmp, _) = FaceMatcherEngine.enhancePortraitWithGeminiOrFallbackBlocking(
                        this@AdminActivity,
                        workingRawSquare
                    )

                    val scaledDp = Bitmap.createScaledBitmap(finalDpBmp, 360, 360, true)
                    val baos = ByteArrayOutputStream()
                    scaledDp.compress(Bitmap.CompressFormat.JPEG, 90, baos)
                    val base64Str = "data:image/jpeg;base64," + Base64.encodeToString(baos.toByteArray(), Base64.NO_WRAP)

                    val escapedPhoto = gson.toJson(base64Str)
                    val escapedEmb = gson.toJson(embedding192D)

                    runOnUiThread {
                        stopKycMonitoring()
                        rootContainer.removeView(overlay)
                        kycOverlayRoot = null
                        webView.evaluateJavascript("onLiveKycPhotoApproved($escapedPhoto, $escapedEmb);", null)
                    }
                }.start()
            }
            .setNegativeButton("दोबारा लें") { _, _ ->
                // दोबारा लेने पर सर्कल को 0% से फिर से शुरू करें
                isKycCaptureLocked = false
                consecutivePerfectKycFrames = 0
                kycRingView?.updateProgressSmooth(0f, false)
                isKycMonitoringActive = true
                kycHandler.postDelayed(kycMonitorRunnable, 200L)
            }
            .create()

        dialog.show()
    }

    override fun onDestroy() {
        super.onDestroy()
        stopKycMonitoring()
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
                    val dim = FaceMatcherEngine.getEmbeddingDimension()
                    for (s in parsed.students) {
                        s.embeddings.removeAll { it.size != dim }
                    }
                    AppStorage.saveData(this@AdminActivity, parsed)
                    FaceMatcherEngine.clearCache()
                    FaceMatcherEngine.warmUpCache(parsed.students, this@AdminActivity)
                }
            } catch (e: Exception) {
                e.printStackTrace()
            }
        }

        @JavascriptInterface
        fun getAiDiagnosticsJson(): String {
            return AiDiagnosticLogger.buildFullDiagnosticJson(this@AdminActivity)
        }

        @JavascriptInterface
        fun triggerLiveAiHealthCheck() {
            Thread {
                val resJson = AiDiagnosticLogger.runLiveApiHealthCheckBlocking(this@AdminActivity)
                val escaped = gson.toJson(resJson)
                runOnUiThread {
                    webView.evaluateJavascript("onLiveAiHealthCheckResult($escaped);", null)
                }
            }.start()
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
                stopKycMonitoring()
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
