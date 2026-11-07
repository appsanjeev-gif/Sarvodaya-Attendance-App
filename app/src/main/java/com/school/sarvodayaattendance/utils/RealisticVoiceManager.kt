package com.school.sarvodayaattendance.utils

import android.content.Context
import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioTrack
import android.media.MediaPlayer
import android.media.audiofx.LoudnessEnhancer
import android.os.Handler
import android.os.Looper
import android.util.Base64
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.io.FileOutputStream
import java.net.HttpURLConnection
import java.net.SocketTimeoutException
import java.net.URL
import java.net.UnknownHostException
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.security.MessageDigest
import java.text.SimpleDateFormat
import java.util.Calendar
import java.util.Date
import java.util.Locale
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.CountDownLatch
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicInteger
import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.exp
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sin
import kotlin.random.Random

object AiDiagnosticLogger {
    private const val PREF_DIAG = "SarvodayaAiDiagPrefs"
    private const val KEY_LOGS = "diag_logs_json"
    private const val KEY_STATUS_MAP = "diag_status_map_json"

    private fun nowTime(): String {
        return SimpleDateFormat("dd.MM hh:mm:ss a", Locale.US).format(Date())
    }

    fun parseElevenLabsErrorDetail(errBody: String, httpCode: Int): String {
        return try {
            val root = JSONObject(errBody)
            val detailObj = root.optJSONObject("detail")
            if (detailObj != null) {
                val status = detailObj.optString("status", "")
                val msg = detailObj.optString("message", "")
                when {
                    status.contains("missing_permissions", ignoreCase = true) ->
                        "Key में आवश्यक परमिशन ($msg) बंद है — ElevenLabs में Key बनाते समय 'Text to Speech' चालू रखें"
                    status.contains("unusual_activity", ignoreCase = true) ->
                        "ElevenLabs ने Free-Tier पर मोबाइल IP/VPN ब्लॉक लगाया है (Google AI Studio TTS पर स्वतः स्विच किया गया)"
                    status.contains("quota_exceeded", ignoreCase = true) ->
                        "ElevenLabs के अक्षरों की लिमिट समाप्त हो गई है (quota_exceeded)"
                    msg.isNotBlank() ->
                        "HTTP $httpCode ($status): ${msg.take(90)}"
                    else ->
                        "HTTP $httpCode: ${errBody.take(80)}"
                }
            } else {
                "HTTP $httpCode: ${errBody.take(80).ifBlank { "अज्ञात सर्वर त्रुटि" }}"
            }
        } catch (e: Exception) {
            if (httpCode == 401) {
                "HTTP 401: Key में Text-to-Speech अनुमति बंद है या गलत Key है"
            } else {
                "HTTP $httpCode: ${errBody.take(65)}"
            }
        }
    }

    @Synchronized
    fun recordEvent(
        context: Context,
        serviceKey: String,
        serviceTitle: String,
        statusBadge: String,
        reasonHindi: String
    ) {
        try {
            val prefs = context.applicationContext.getSharedPreferences(PREF_DIAG, Context.MODE_PRIVATE)
            val timeStr = nowTime()

            val statusRaw = prefs.getString(KEY_STATUS_MAP, "{}") ?: "{}"
            val statusObj = try { JSONObject(statusRaw) } catch (e: Exception) { JSONObject() }
            statusObj.put(serviceKey, JSONObject().apply {
                put("title", serviceTitle)
                put("status", statusBadge)
                put("reason", reasonHindi)
                put("time", timeStr)
            })

            val logsRaw = prefs.getString(KEY_LOGS, "[]") ?: "[]"
            val oldArr = try { JSONArray(logsRaw) } catch (e: Exception) { JSONArray() }
            val newArr = JSONArray()
            newArr.put(JSONObject().apply {
                put("time", timeStr)
                put("service", serviceTitle)
                put("status", statusBadge)
                put("reason", reasonHindi)
            })
            val limit = minOf(oldArr.length(), 24)
            for (i in 0 until limit) {
                newArr.put(oldArr.getJSONObject(i))
            }

            prefs.edit()
                .putString(KEY_STATUS_MAP, statusObj.toString())
                .putString(KEY_LOGS, newArr.toString())
                .apply()
        } catch (e: Exception) {}
    }

    fun buildFullDiagnosticJson(context: Context): String {
        return try {
            val appCtx = context.applicationContext
            val prefs = appCtx.getSharedPreferences(PREF_DIAG, Context.MODE_PRIVATE)
            val statusObj = try { JSONObject(prefs.getString(KEY_STATUS_MAP, "{}") ?: "{}") } catch (e: Exception) { JSONObject() }
            val logsArr = try { JSONArray(prefs.getString(KEY_LOGS, "[]") ?: "[]") } catch (e: Exception) { JSONArray() }

            val vm = RealisticVoiceManager(appCtx)
            val gKeys = vm.getGeminiKeysListPublic()
            val eKeys = vm.getElevenLabsKeysListPublic()

            val root = JSONObject().apply {
                put("checkedAt", nowTime())
                put("tfliteReady", FaceMatcherEngine.isNeuralModelReady())
                put("geminiKeyCount", gKeys.size)
                put("elevenKeyCount", eKeys.size)
                put("services", statusObj)
                put("logs", logsArr)
            }
            root.toString()
        } catch (e: Exception) {
            "{}"
        }
    }

    fun runLiveApiHealthCheckBlocking(context: Context): String {
        val appCtx = context.applicationContext
        val vm = RealisticVoiceManager(appCtx)
        val gKeys = vm.getGeminiKeysListPublic()
        val eKeys = vm.getElevenLabsKeysListPublic()

        val geminiResults = JSONArray()
        val testModels = listOf("gemini-2.0-flash", "gemini-1.5-flash", "gemini-2.5-flash")

        for (k in gKeys) {
            val cleanK = k.trim()
            val masked = if (cleanK.length > 10) "${cleanK.take(6)}••••••••${cleanK.takeLast(4)}" else cleanK
            var status = "❌ अमान्य"
            var detail = "कोई रिस्पॉन्स नहीं"
            var lastCode = 0
            var lastErr = ""

            for (mName in testModels) {
                try {
                    val url = "https://generativelanguage.googleapis.com/v1beta/models/$mName:generateContent?key=$cleanK"
                    val conn = (URL(url).openConnection() as HttpURLConnection).apply {
                        requestMethod = "POST"
                        connectTimeout = 5000
                        readTimeout = 6000
                        doOutput = true
                        setRequestProperty("Content-Type", "application/json; charset=UTF-8")
                    }
                    val body = JSONObject().put("contents", JSONArray().put(JSONObject().put("parts", JSONArray().put(JSONObject().put("text", "Reply OK")))))
                    conn.outputStream.use { it.write(body.toString().toByteArray(Charsets.UTF_8)) }
                    lastCode = conn.responseCode
                    if (lastCode == 200) {
                        status = "✅ Key चालू है (Active)"
                        detail = "HTTP 200 • मॉडल ($mName) + Gemini TTS तैयार है"
                        recordEvent(appCtx, "SLOGAN", "5. दैनिक AI स्लोगन", "✅ सक्रिय ($mName)", "Gemini API Key लाइव जांच में सफल")
                        break
                    } else {
                        lastErr = try { conn.errorStream?.bufferedReader()?.readText() ?: "" } catch (e: Exception) { "" }
                        if (lastCode != 404) break
                    }
                } catch (e: UnknownHostException) {
                    lastCode = -1
                    status = "❌ इंटरनेट बंद है"
                    detail = "मोबाइल डेटा या वाई-फाई चालू नहीं है"
                    break
                } catch (e: SocketTimeoutException) {
                    lastCode = -2
                    status = "⚠️ नेटवर्क धीमा"
                    detail = "इंटरनेट बहुत स्लो है (Connection Timeout)"
                    break
                } catch (e: Exception) {
                    lastCode = -3
                    status = "❌ कनेक्शन त्रुटि"
                    detail = e.javaClass.simpleName
                    break
                }
            }

            if (lastCode == 404) {
                try {
                    val listUrl = "https://generativelanguage.googleapis.com/v1beta/models?key=$cleanK"
                    val listConn = (URL(listUrl).openConnection() as HttpURLConnection).apply {
                        requestMethod = "GET"
                        connectTimeout = 5000
                        readTimeout = 5000
                    }
                    if (listConn.responseCode == 200) {
                        val listResp = listConn.inputStream.bufferedReader(Charsets.UTF_8).use { it.readText() }
                        val modelsArr = JSONObject(listResp).optJSONArray("models")
                        val modelCount = modelsArr?.length() ?: 0
                        status = "✅ Key वैध है (Models OK • $modelCount उपलब्ध)"
                        detail = "Key सक्रिय है (उपलब्ध मॉडल्स: $modelCount)"
                    } else {
                        status = "❌ त्रुटि (HTTP 404)"
                        detail = "इस Key पर जनरेशन मॉडल उपलब्ध नहीं है"
                    }
                } catch (e: Exception) {}
            } else if (lastCode > 0 && lastCode != 200) {
                status = "❌ त्रुटि (HTTP $lastCode)"
                detail = when (lastCode) {
                    400 -> "Key गलत है या निष्क्रिय है (Invalid Key)"
                    403 -> "इस Key पर अनुमति नहीं है (Forbidden)"
                    429 -> "आज की फ्री लिमिट समाप्त (Quota Exceeded 429)"
                    else -> "सर्वर कोड $lastCode: ${lastErr.take(60)}"
                }
            }

            geminiResults.put(JSONObject().apply {
                put("maskedKey", masked)
                put("status", status)
                put("detail", detail)
            })
        }

        val elevenResults = JSONArray()
        for (k in eKeys) {
            val cleanKey = k.trim()
            val masked = if (cleanKey.length > 10) "${cleanKPreview(cleanKey)}" else cleanKey
            var status = "❌ अमान्य"
            var detail = ""
            try {
                val url = "https://api.elevenlabs.io/v1/user/subscription"
                val conn = (URL(url).openConnection() as HttpURLConnection).apply {
                    requestMethod = "GET"
                    connectTimeout = 6000
                    readTimeout = 7000
                    setRequestProperty("xi-api-key", cleanKey)
                    setRequestProperty("Accept", "application/json")
                    setRequestProperty("User-Agent", "Mozilla/5.0 (Linux; Android 14) AppleWebKit/537.36 Chrome/126.0.0.0 Mobile")
                }
                val code = conn.responseCode
                if (code == 200) {
                    val resp = conn.inputStream.bufferedReader(Charsets.UTF_8).use { it.readText() }
                    val sub = JSONObject(resp)
                    val tier = sub.optString("tier", "free").uppercase()
                    val used = sub.optInt("character_count", 0)
                    val limit = sub.optInt("character_limit", 10000)
                    val rem = (limit - used).coerceAtLeast(0)
                    val nextUnix = sub.optLong("next_character_count_reset_unix", 0L)
                    val resetDate = if (nextUnix > 0L) {
                        SimpleDateFormat("dd MMM yyyy", Locale.US).format(Date(nextUnix * 1000L))
                    } else "-"
                    status = "✅ Key चालू है (Active • Tier: $tier)"
                    detail = "बची हुई लिमिट: $rem / $limit chars (Used: $used) | अगली Reset: $resetDate"
                    recordEvent(appCtx, "VOICE", "4. AI मानवीय आवाज़ (TTS)", "✅ ElevenLabs सक्रिय", "बचे हुए अक्षर: $rem / $limit")
                } else {
                    val errBody = try { conn.errorStream?.bufferedReader()?.readText() ?: "" } catch (e: Exception) { "" }
                    if (code == 401 && errBody.contains("user_read", ignoreCase = true)) {
                        val ttsUrl = "https://api.elevenlabs.io/v1/text-to-speech/21m00Tcm4TlvDq8ikWAM"
                        val ttsConn = (URL(ttsUrl).openConnection() as HttpURLConnection).apply {
                            requestMethod = "POST"
                            connectTimeout = 7000
                            readTimeout = 9000
                            doOutput = true
                            setRequestProperty("xi-api-key", cleanKey)
                            setRequestProperty("Content-Type", "application/json; charset=UTF-8")
                            setRequestProperty("Accept", "audio/mpeg")
                            setRequestProperty("User-Agent", "Mozilla/5.0 (Linux; Android 14) AppleWebKit/537.36 Chrome/126.0.0.0 Mobile")
                        }
                        val testPayload = JSONObject().apply {
                            put("text", "नमस्ते")
                            put("model_id", "eleven_multilingual_v2")
                        }
                        ttsConn.outputStream.use { it.write(testPayload.toString().toByteArray(Charsets.UTF_8)) }
                        val ttsCode = ttsConn.responseCode
                        if (ttsCode == 200) {
                            status = "✅ Key चालू है (TTS Active • Restricted Key)"
                            detail = "ऑडियो जनरेशन चालू है (Subscription Read परमिशन बंद होने से कोटा संख्या छिपी है)"
                            recordEvent(appCtx, "VOICE", "4. AI मानवीय आवाज़ (TTS)", "✅ ElevenLabs सक्रिय", "Restricted Key से ऑडियो जनरेशन चालू है")
                        } else {
                            val ttsErr = try { ttsConn.errorStream?.bufferedReader()?.readText() ?: "" } catch (e: Exception) { "" }
                            status = "❌ त्रुटि (HTTP $ttsCode)"
                            detail = parseElevenLabsErrorDetail(ttsErr, ttsCode)
                        }
                    } else {
                        status = "❌ त्रुटि (HTTP $code)"
                        detail = parseElevenLabsErrorDetail(errBody, code)
                    }
                }
            } catch (e: UnknownHostException) {
                status = "❌ इंटरनेट बंद है"
                detail = "मोबाइल डेटा या वाई-फाई चालू नहीं है"
            } catch (e: SocketTimeoutException) {
                status = "⚠️ नेटवर्क धीमा"
                detail = "इंटरनेट बहुत स्लो है (Timeout)"
            } catch (e: Exception) {
                status = "❌ कनेक्शन त्रुटि"
                detail = e.javaClass.simpleName
            }
            elevenResults.put(JSONObject().apply {
                put("maskedKey", masked)
                put("status", status)
                put("detail", detail)
            })
        }

        val fullJson = JSONObject(buildFullDiagnosticJson(appCtx))
        fullJson.put("geminiLiveCheck", geminiResults)
        fullJson.put("elevenLiveCheck", elevenResults)
        return fullJson.toString()
    }

    private fun cleanKPreview(k: String): String {
        return "${k.take(6)}••••••••${k.takeLast(4)}"
    }
}

class RealisticVoiceManager(private val context: Context) {

    companion object {
        // जो API Key ब्लॉक या कोटा-समाप्त हो जाए, उसे 8 मिनट के लिए कूलडाउन में रखें ताकि अगली आवाज़ 0 सेकंड में दूसरी चालू Key से बने
        private val badKeyCooldownUntilMs = ConcurrentHashMap<String, Long>()
        // यदि किसी AQ.Ab8 Key के लिए हमें मॉडल लिस्ट से सही मॉडल नाम मिल जाए तो उसे याद रखें
        private val discoveredModelForKey = ConcurrentHashMap<String, String>()
    }

    private val appCtx = context.applicationContext
    private val prefs = appCtx.getSharedPreferences("SarvodayaVoicePrefs", Context.MODE_PRIVATE)
    private val mainHandler = Handler(Looper.getMainLooper())

    private val geminiKeyIndex = AtomicInteger(0)
    private val elevenKeyIndex = AtomicInteger(0)

    fun getGeminiKeysString(): String {
        val dedicated = prefs.getString("custom_gemini_only_keys", null)
        if (!dedicated.isNullOrBlank()) return dedicated
        val legacy = prefs.getString("custom_gemini_api_key", "") ?: ""
        return legacy.split(",", "\n", ";", " ")
            .map { it.trim() }
            .filter { it.length > 15 && (it.startsWith("AIza") || it.startsWith("AQ.")) }
            .joinToString("\n")
    }

    fun getElevenLabsKeysString(): String {
        val dedicated = prefs.getString("custom_eleven_only_keys", null)
        if (!dedicated.isNullOrBlank()) return dedicated
        val legacy = prefs.getString("custom_gemini_api_key", "") ?: ""
        return legacy.split(",", "\n", ";", " ")
            .map { it.trim() }
            .filter { it.startsWith("sk_", ignoreCase = true) || (it.length >= 28 && !it.startsWith("AIza") && !it.startsWith("AQ.")) }
            .joinToString("\n")
    }

    fun getGeminiKeysListPublic(): List<String> = getGeminiKeyPool()
    fun getElevenLabsKeysListPublic(): List<String> = getElevenLabsKeyPool()

    fun saveSeparateApiKeys(geminiRaw: String, elevenRaw: String) {
        val gClean = geminiRaw.split(",", "\n", ";", " ")
            .map { it.trim() }
            .filter { it.length > 15 && !it.startsWith("sk_", ignoreCase = true) }
            .joinToString("\n")

        val eClean = elevenRaw.split(",", "\n", ";", " ")
            .map { it.trim() }
            .filter { it.length > 15 && !it.startsWith("AIza") && !it.startsWith("AQ.") }
            .joinToString("\n")

        val combined = listOf(gClean, eClean).filter { it.isNotEmpty() }.joinToString("\n")

        prefs.edit()
            .putString("custom_gemini_only_keys", gClean)
            .putString("custom_eleven_only_keys", eClean)
            .putString("custom_gemini_api_key", combined)
            .apply()

        badKeyCooldownUntilMs.clear()

        AiDiagnosticLogger.recordEvent(
            appCtx,
            "VOICE",
            "4. AI मानवीय आवाज़ (TTS)",
            "🔄 Keys अपडेट की गईं",
            "Gemini Keys: ${if (gClean.isEmpty()) 0 else gClean.lines().size}, ElevenLabs Keys: ${if (eClean.isEmpty()) 0 else eClean.lines().size}"
        )

        rollingCacheExecutor.execute {
            restoreAndCleanCacheDirectory()
            maintainTwoHourRollingWelcomeCache()
        }
    }

    fun getApiKey(): String {
        val combined = listOf(getGeminiKeysString().trim(), getElevenLabsKeysString().trim())
            .filter { it.isNotEmpty() }
            .joinToString("\n")
        if (combined.isNotEmpty()) return combined
        return prefs.getString("custom_gemini_api_key", "") ?: ""
    }

    fun saveCustomApiKey(newKey: String) {
        val tokens = newKey.split(",", "\n", ";", " ").map { it.trim() }.filter { it.length > 15 }
        val gKeys = tokens.filter { it.startsWith("AIza") || it.startsWith("AQ.") }.joinToString("\n")
        val eKeys = tokens.filter { !it.startsWith("AIza") && !it.startsWith("AQ.") }.joinToString("\n")
        saveSeparateApiKeys(gKeys, eKeys)
    }

    // सभी Google Keys (AIza... और AQ.Ab8...) को शामिल करें (AIza को प्राथमिकता दें, लेकिन किसी भी Key को ब्लॉक न करें!)
    private fun getGeminiKeyPool(): List<String> {
        val raw = getGeminiKeysString()
        val all = raw.split(",", "\n", ";", " ")
            .map { it.trim() }
            .filter { it.length > 15 && !it.startsWith("sk_", ignoreCase = true) }
            .distinct()
        val aizaFirst = all.filter { it.startsWith("AIza") }
        val others = all.filter { !it.startsWith("AIza") }
        return aizaFirst + others
    }

    private fun getElevenLabsKeyPool(): List<String> {
        val raw = getElevenLabsKeysString()
        return raw.split(",", "\n", ";", " ")
            .map { it.trim() }
            .filter { it.length > 15 && !it.startsWith("AIza") && !it.startsWith("AQ.") }
            .distinct()
    }

    private fun isKeyAvailableNow(key: String): Boolean {
        val until = badKeyCooldownUntilMs[key] ?: return true
        if (System.currentTimeMillis() >= until) {
            badKeyCooldownUntilMs.remove(key)
            return true
        }
        return false
    }

    private fun markKeyCooldown(key: String, durationMs: Long) {
        badKeyCooldownUntilMs[key] = System.currentTimeMillis() + durationMs
    }

    private data class AiVoiceProfile(
        val name: String,
        val elevenVoiceIds: List<String>,
        val stylePrompt: String
    )

    private val aiVoiceProfiles = listOf(
        AiVoiceProfile(
            name = "Sadachbia",
            elevenVoiceIds = listOf(
                "21m00Tcm4TlvDq8ikWAM", // Rachel
                "pNInz6obpgDQGcFmaJgB"  // Adam
            ),
            stylePrompt = "Speak like an energetic, proud, warm Indian Hindi radio RJ host with a natural smile and lively pace:"
        ),
        AiVoiceProfile(
            name = "Autonoe",
            elevenVoiceIds = listOf(
                "21m00Tcm4TlvDq8ikWAM", // Rachel
                "EXAVITQu4vr4xnSDxMaL"  // Sarah
            ),
            stylePrompt = "Speak like a cheerful, bright, inspiring Indian Hindi female radio RJ announcer with warmth and pride:"
        )
    )

    var isVoiceEnabled = true

    @Volatile
    var isCurrentlySpeaking: Boolean = false
        private set

    private var unlockRunnable: Runnable? = null
    private val voiceExecutor = Executors.newSingleThreadExecutor()
    private val rollingCacheExecutor = Executors.newSingleThreadExecutor()
    private var currentAudioTrack: AudioTrack? = null
    private var loudnessEnhancer: LoudnessEnhancer? = null
    private var mediaPlayer: MediaPlayer? = null

    // मुख्य स्थायी कैश फोल्डर (जिसमें छात्रों की एक बार बनी आवाज़ हमेशा सुरक्षित रहती है)
    private val voiceCacheDir: File by lazy {
        File(appCtx.getExternalFilesDir(null) ?: appCtx.filesDir, "ai_studio_voice_bank").apply {
            if (!exists()) mkdirs()
        }
    }

    init {
        rollingCacheExecutor.execute {
            restoreAndCleanCacheDirectory()
            // ऐप खुलने के 8 सेकंड बाद बैकग्राउंड में वेलकम कैश अपडेट करें ताकि स्टार्टअप पर कोई लोड न पड़े
            Thread.sleep(8000L)
            maintainTwoHourRollingWelcomeCache()
        }
    }

    private fun isValidPcmAudioBytes(bytes: ByteArray): Boolean {
        if (bytes.size < 3600) return false
        return try {
            val buf = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
            var peak = 0
            var nonZeroSamples = 0
            val totalSamples = bytes.size / 2
            val step = max(1, totalSamples / 300)
            var idx = 0
            while (idx < totalSamples) {
                val s = abs(buf.getShort(idx * 2).toInt())
                if (s > peak) peak = s
                if (s > 120) nonZeroSamples++
                idx += step
            }
            peak > 450 && nonZeroSamples > 8
        } catch (e: Exception) {
            false
        }
    }

    // किसी भी वैध कैश ऑडियो को कभी डिलीट न करें; यदि v2 फोल्डर में कोई ऑडियो बना हो तो उसे भी मुख्य फोल्डर में वापस ले आएं
    private fun restoreAndCleanCacheDirectory() {
        try {
            val v2Dir = File(appCtx.getExternalFilesDir(null) ?: appCtx.filesDir, "ai_studio_voice_bank_v2")
            if (v2Dir.exists()) {
                v2Dir.listFiles()?.forEach { f ->
                    if (f.length() > 1500L) {
                        val dest = File(voiceCacheDir, f.name)
                        if (!dest.exists()) {
                            try { f.copyTo(dest, overwrite = false) } catch (e: Exception) {}
                        }
                    }
                }
            }

            voiceCacheDir.listFiles()?.forEach { file ->
                if (file.name.endsWith(".pcm") && file.length() < 3600L) {
                    try { file.delete() } catch (e: Exception) {}
                } else if (file.name.endsWith(".mp3") && file.length() < 1000L) {
                    try { file.delete() } catch (e: Exception) {}
                }
            }
        } catch (e: Exception) {}
    }

    private fun getCurrentTwoHourSlot(): Long {
        return System.currentTimeMillis() / (2L * 60L * 60L * 1000L)
    }

    private fun maintainTwoHourRollingWelcomeCache() {
        rollingCacheExecutor.execute {
            try {
                val currentSlot = getCurrentTwoHourSlot()
                val nextSlot = currentSlot + 1L

                if (!isSlotAudioReady(currentSlot)) {
                    generateAndCacheSlotAudio(currentSlot, isNextSlot = false)
                }

                if (!isSlotAudioReady(nextSlot)) {
                    generateAndCacheSlotAudio(nextSlot, isNextSlot = true)
                }

                // केवल तब पुराने स्लॉट हटाएं जब वर्तमान स्लॉट का ऑडियो तैयार हो चुका हो
                if (isSlotAudioReady(currentSlot)) {
                    voiceCacheDir.listFiles()?.forEach { file ->
                        if (file.name.startsWith("welcome_slot_")) {
                            val slotNum = file.name.removePrefix("welcome_slot_")
                                .substringBefore(".")
                                .toLongOrNull()
                            if (slotNum == null || (slotNum != currentSlot && slotNum != nextSlot)) {
                                try { file.delete() } catch (e: Exception) {}
                            }
                        }
                    }
                }
            } catch (e: Exception) {}
        }
    }

    private fun isSlotAudioReady(slotId: Long): Boolean {
        val mp3 = File(voiceCacheDir, "welcome_slot_$slotId.mp3")
        if (mp3.exists() && mp3.length() > 1200) return true
        val pcm = File(voiceCacheDir, "welcome_slot_$slotId.pcm")
        if (pcm.exists() && pcm.length() > 3600) {
            val bytes = try { pcm.readBytes() } catch (e: Exception) { ByteArray(0) }
            if (isValidPcmAudioBytes(bytes)) return true
            try { pcm.delete() } catch (e: Exception) {}
        }
        return false
    }

    private fun generateAndCacheSlotAudio(slotId: Long, isNextSlot: Boolean): Boolean {
        val welcomeText = generateFreshAiWelcomeLineForSlot(slotId, isNextSlot)
        val profile = aiVoiceProfiles[(slotId % aiVoiceProfiles.size).toInt().let { if (it < 0) -it else it }]
        val targetPcm = File(voiceCacheDir, "welcome_slot_$slotId.pcm")
        val targetMp3 = File(voiceCacheDir, "welcome_slot_$slotId.mp3")

        // 1. पहले सभी 10+ Google Gemini Keys से प्रयास करें (क्योंकि आपके पास 10+ Google Keys हैं)
        val geminiPool = getGeminiKeyPool()
        if (geminiPool.isNotEmpty()) {
            val activeGemini = geminiPool.filter { isKeyAvailableNow(it) }.ifEmpty { geminiPool }
            for (attempt in activeGemini.indices) {
                val idx = (geminiKeyIndex.getAndIncrement() % activeGemini.size).let { if (it < 0) it + activeGemini.size else it }
                val candidateKey = activeGemini[idx]
                val (pcmBytes, code, errReason) = fetchGoogleAiStudioTtsPcm(welcomeText, profile, candidateKey)
                if (pcmBytes != null && isValidPcmAudioBytes(pcmBytes)) {
                    writeFileAtomically(targetPcm, pcmBytes)
                    AiDiagnosticLogger.recordEvent(appCtx, "VOICE", "4. AI मानवीय आवाज़ (TTS)", "✅ Gemini TTS सक्रिय", "Google AI Studio से स्वागत ऑडियो तैयार हुआ")
                    return true
                } else {
                    AiDiagnosticLogger.recordEvent(appCtx, "VOICE", "4. AI मानवीय आवाज़ (TTS)", "⚠️ Gemini TTS प्रयास", errReason)
                    if (code == -1) break // इंटरनेट बंद होने पर तुरंत रुकें
                    if (code == 400 || code == 403 || code == 404 || code == 429) {
                        markKeyCooldown(candidateKey, 5 * 60 * 1000L)
                        continue // अगली Google Key आज़माएं!
                    }
                }
            }
        }

        // 2. फिर ElevenLabs Keys आज़माएं
        val elevenPool = getElevenLabsKeyPool()
        if (elevenPool.isNotEmpty()) {
            val activeEleven = elevenPool.filter { isKeyAvailableNow(it) }
            for (attempt in activeEleven.indices) {
                val idx = (elevenKeyIndex.getAndIncrement() % activeEleven.size).let { if (it < 0) it + activeEleven.size else it }
                val candidateKey = activeEleven[idx]
                val (mp3Bytes, code, errReason) = fetchElevenLabsMp3WithFallback(welcomeText, profile.elevenVoiceIds, candidateKey)
                if (mp3Bytes != null && mp3Bytes.size > 1200) {
                    writeFileAtomically(targetMp3, mp3Bytes)
                    AiDiagnosticLogger.recordEvent(appCtx, "VOICE", "4. AI मानवीय आवाज़ (TTS)", "✅ ElevenLabs सक्रिय", "ElevenLabs से स्वागत ऑडियो तैयार हुआ")
                    return true
                } else {
                    AiDiagnosticLogger.recordEvent(appCtx, "VOICE", "4. AI मानवीय आवाज़ (TTS)", "❌ ElevenLabs विफल", errReason)
                    if (code == -1) break
                    if (code == 401 || code == 402 || code == 403 || code == 429) {
                        markKeyCooldown(candidateKey, 10 * 60 * 1000L)
                        continue
                    }
                }
            }
        }

        return false
    }

    private fun generateFreshAiWelcomeLineForSlot(slotId: Long, isNextSlot: Boolean): String {
        val cal = Calendar.getInstance()
        if (isNextSlot) cal.add(Calendar.HOUR_OF_DAY, 2)
        val hour = cal.get(Calendar.HOUR_OF_DAY)
        val greetingWord = when {
            hour < 12 -> "सुप्रभात"
            hour < 17 -> "नमस्कार"
            else -> "शुभ संध्या"
        }

        val styles = listOf(
            "प्रेरणादायक और जोशपूर्ण (Motivational & Energetic)",
            "हाई-टेक डिजिटल स्मार्ट कैंपस (Modern Digital Campus)",
            "बच्चों और शिक्षकों से आत्मीय संवाद (Warm Student-Teacher Connect)",
            "अनुशासन, मुस्कान और गर्व से भरा (Proud & Disciplined)"
        )
        val chosenStyle = styles[(slotId % styles.size).toInt().let { if (it < 0) -it else it }]

        val geminiPool = getGeminiKeyPool().filter { isKeyAvailableNow(it) }
        val sloganModels = listOf("gemini-2.0-flash", "gemini-1.5-flash", "gemini-2.5-flash")

        for (apiKey in geminiPool.take(3)) {
            for (modelName in sloganModels) {
                try {
                    val endpoint = "https://generativelanguage.googleapis.com/v1beta/models/$modelName:generateContent?key=$apiKey"
                    val conn = (URL(endpoint).openConnection() as HttpURLConnection).apply {
                        requestMethod = "POST"
                        connectTimeout = 4500
                        readTimeout = 5500
                        doOutput = true
                        setRequestProperty("Content-Type", "application/json; charset=UTF-8")
                    }

                    val prompt = """
                        विद्यालय का नाम: "सर्वोदय मध्य विद्यालय, सोहसराय"।
                        अभिवादन शब्द: "$greetingWord"।
                        अंदाज़: $chosenStyle।
                        नियम: केवल 18 से 24 शब्दों में एक नया, आकर्षक रेडियो एंकर जैसा हिंदी स्वागत वाक्य लिखें जो छात्रों को कैमरे के सामने मुस्कुराकर स्मार्ट उपस्थिति दर्ज कराने के लिए प्रेरित करे। केवल शुद्ध वाक्य लौटाएं, कोई कोट्स या अतिरिक्त शब्द न लिखें। (Slot: $slotId)
                    """.trimIndent()

                    val payload = JSONObject().apply {
                        put("contents", JSONArray().put(
                            JSONObject().put("parts", JSONArray().put(JSONObject().put("text", prompt)))
                        ))
                    }

                    conn.outputStream.use { it.write(payload.toString().toByteArray(Charsets.UTF_8)) }
                    val code = conn.responseCode
                    if (code == 200) {
                        val resp = conn.inputStream.bufferedReader(Charsets.UTF_8).use { it.readText() }
                        val text = JSONObject(resp)
                            .optJSONArray("candidates")?.optJSONObject(0)
                            ?.optJSONObject("content")?.optJSONArray("parts")?.optJSONObject(0)
                            ?.optString("text", "")?.trim()
                            ?.replace("\"", "")?.replace("*", "")?.replace("\n", " ") ?: ""
                        if (text.length in 25..220 && text.contains("सर्वोदय")) {
                            AiDiagnosticLogger.recordEvent(appCtx, "SLOGAN", "5. दैनिक AI स्लोगन", "✅ सफल ($modelName)", "नया स्लोगन: \"${text.take(45)}...\"")
                            return text
                        }
                    } else if (code != 404) {
                        break
                    }
                } catch (e: UnknownHostException) {
                    break
                } catch (e: Exception) {
                    break
                }
            }
        }

        AiDiagnosticLogger.recordEvent(appCtx, "SLOGAN", "5. दैनिक AI स्लोगन", "ℹ️ इन-बिल्ट स्लोगन", "इन-बिल्ट हिंदी स्वागत स्लोगन चुना गया")

        val fallbackTemplates = listOf(
            "$greetingWord! ज्ञान के मंदिर सर्वोदय मध्य विद्यालय, सोहसराय में आपका हार्दिक स्वागत है। नई ऊर्जा के साथ आइए आज की स्मार्ट उपस्थिति शुरू करें!",
            "$greetingWord! सर्वोदय मध्य विद्यालय, सोहसराय के हाई-टेक डिजिटल कैंपस में आपका अभिनंदन है। हमारा एआई कैमरा तैयार है, कृपया मुस्कुराते हुए अपनी हाज़िरी लगाएं!",
            "$greetingWord बच्चों और गुरुजनों! सर्वोदय मध्य विद्यालय, सोहसराय में एक और शानदार दिन की शुरुआत हो चुकी है। आइए, कैमरे के सामने आकर उपस्थिति दर्ज कराएं।",
            "$greetingWord! शिक्षा और संस्कार के केंद्र सर्वोदय मध्य विद्यालय, सोहसराय में आपका स्वागत है। स्मार्ट अटेंडेंस सिस्टम अब लाइव है, चलिए आज का रिकॉर्ड बनाएं!"
        )
        return fallbackTemplates[(slotId % fallbackTemplates.size).toInt().let { if (it < 0) -it else it }]
    }

    private fun writeFileAtomically(targetFile: File, bytes: ByteArray) {
        try {
            val tempFile = File(targetFile.parentFile, "${targetFile.name}.tmp")
            FileOutputStream(tempFile).use { fos ->
                fos.write(bytes)
                fos.fd.sync()
            }
            if (tempFile.exists() && tempFile.length() > 1000) {
                if (targetFile.exists()) targetFile.delete()
                tempFile.renameTo(targetFile)
            }
        } catch (e: Exception) {}
    }

    private fun scheduleSpeakingUnlock(durationMs: Long) {
        isCurrentlySpeaking = true
        unlockRunnable?.let { mainHandler.removeCallbacks(it) }
        val r = Runnable { isCurrentlySpeaking = false }
        unlockRunnable = r
        mainHandler.postDelayed(r, durationMs)
    }

    private fun playSoftStartupChimeBlocking() {
        try {
            val sampleRate = 24000
            val durationMs = 420
            val totalSamples = (sampleRate * durationMs) / 1000
            val pcmBytes = ByteArray(totalSamples * 2)
            val buf = ByteBuffer.wrap(pcmBytes).order(ByteOrder.LITTLE_ENDIAN)

            for (i in 0 until totalSamples) {
                val t = i.toDouble() / sampleRate
                val freq = if (i < totalSamples / 2) 587.33 else 880.0
                val env = exp(-4.5 * (i.toDouble() / totalSamples))
                val wave = sin(2.0 * PI * freq * t) * env * 0.38
                val sample = (wave * Short.MAX_VALUE).toInt().coerceIn(Short.MIN_VALUE.toInt(), Short.MAX_VALUE.toInt())
                buf.putShort(sample.toShort())
            }

            val minBuf = AudioTrack.getMinBufferSize(sampleRate, AudioFormat.CHANNEL_OUT_MONO, AudioFormat.ENCODING_PCM_16BIT)
            val track = AudioTrack.Builder()
                .setAudioAttributes(
                    AudioAttributes.Builder()
                        .setUsage(AudioAttributes.USAGE_MEDIA)
                        .setContentType(AudioAttributes.CONTENT_TYPE_SONIFICATION)
                        .build()
                )
                .setAudioFormat(
                    AudioFormat.Builder()
                        .setEncoding(AudioFormat.ENCODING_PCM_16BIT)
                        .setSampleRate(sampleRate)
                        .setChannelMask(AudioFormat.CHANNEL_OUT_MONO)
                        .build()
                )
                .setBufferSizeInBytes(max(minBuf, 8192))
                .setTransferMode(AudioTrack.MODE_STREAM)
                .build()

            track.play()
            track.write(pcmBytes, 0, pcmBytes.size)
            Thread.sleep(durationMs.toLong() + 30L)
            track.stop()
            track.release()
        } catch (e: Exception) {}
    }

    fun speakStartupWelcome() {
        if (!isVoiceEnabled) return
        voiceExecutor.execute {
            val currentSlot = getCurrentTwoHourSlot()
            val slotMp3 = File(voiceCacheDir, "welcome_slot_$currentSlot.mp3")
            val slotPcm = File(voiceCacheDir, "welcome_slot_$currentSlot.pcm")

            if (slotMp3.exists() && slotMp3.length() > 1200) {
                playSoftStartupChimeBlocking()
                playMp3FileBlocking(slotMp3)
                return@execute
            }
            if (slotPcm.exists() && slotPcm.length() > 3600) {
                val bytes = try { slotPcm.readBytes() } catch (e: Exception) { ByteArray(0) }
                if (isValidPcmAudioBytes(bytes)) {
                    playSoftStartupChimeBlocking()
                    playRawPcm24kHzBlocking(bytes, 1.55f)
                    return@execute
                }
            }

            // यदि वर्तमान स्लॉट अभी बन रहा है, तो क्या कोई पिछला सेव किया हुआ वेलकम स्लॉट मौजूद है? यदि हाँ तो तुरंत उसे चलाएं और बैकग्राउंड में नया बनाएं!
            val anySavedWelcomeMp3 = voiceCacheDir.listFiles()?.firstOrNull { it.name.startsWith("welcome_slot_") && it.name.endsWith(".mp3") && it.length() > 1200 }
            if (anySavedWelcomeMp3 != null) {
                playSoftStartupChimeBlocking()
                playMp3FileBlocking(anySavedWelcomeMp3)
                maintainTwoHourRollingWelcomeCache()
                return@execute
            }
            val anySavedWelcomePcm = voiceCacheDir.listFiles()?.firstOrNull { it.name.startsWith("welcome_slot_") && it.name.endsWith(".pcm") && it.length() > 3600 }
            if (anySavedWelcomePcm != null) {
                val bytes = try { anySavedWelcomePcm.readBytes() } catch (e: Exception) { ByteArray(0) }
                if (isValidPcmAudioBytes(bytes)) {
                    playSoftStartupChimeBlocking()
                    playRawPcm24kHzBlocking(bytes, 1.55f)
                    maintainTwoHourRollingWelcomeCache()
                    return@execute
                }
            }

            if (generateAndCacheSlotAudio(currentSlot, isNextSlot = false)) {
                if (slotMp3.exists() && slotMp3.length() > 1200) {
                    playSoftStartupChimeBlocking()
                    playMp3FileBlocking(slotMp3)
                } else if (slotPcm.exists() && slotPcm.length() > 3600) {
                    playSoftStartupChimeBlocking()
                    playRawPcm24kHzBlocking(slotPcm.readBytes(), 1.55f)
                }
            }
        }
    }

    fun speakVipInspectionWelcome(presentCount: Int, totalCount: Int) {
        if (!isVoiceEnabled) return
        val safeTotal = totalCount.coerceAtLeast(0)
        val safePresent = presentCount.coerceIn(0, safeTotal)

        val msg = when {
            safeTotal == 0 -> {
                "नमस्कार! सर्वोदय मध्य विद्यालय, सोहसराय के स्मार्ट एआई परिसर में आपका हार्दिक अभिनंदन है। अभी सिस्टम में कोई छात्र पंजीकृत नहीं है।"
            }
            safePresent == 0 -> {
                "नमस्कार! सर्वोदय मध्य विद्यालय, सोहसराय के स्मार्ट एआई परिसर में माननीय अधिकारी एवं अतिथि महोदय का हार्दिक अभिनंदन है। आज विद्यालय के कुल $safeTotal छात्रों में से अभी तक किसी भी छात्र की उपस्थिति दर्ज नहीं हुई है।"
            }
            else -> {
                "नमस्कार! सर्वोदय मध्य विद्यालय, सोहसराय के स्मार्ट एआई परिसर में माननीय अधिकारी एवं अतिथि महोदय का हार्दिक अभिनंदन है। आज विद्यालय में कुल $safeTotal में से $safePresent छात्र उपस्थित हैं।"
            }
        }
        speakHindi(msg, boostFactor = 1.65f)
    }

    fun speakGreeting(type: String, studentName: String) {
        if (!isVoiceEnabled) return
        val cleanName = studentName.trim()
        val firstName = cleanName.substringBefore(" ").ifBlank { cleanName }

        val message = when (type) {
            "success" -> "स्वागत है $cleanName! आपकी आज की उपस्थिति दर्ज हो चुकी है।"
            "warning" -> "ध्यान दें $cleanName, आपकी आज की हाज़िरी पहले ही लग चुकी है।"
            "move_away" -> "अरे वाह $firstName, आज यहीं खड़े रहने का इरादा है क्या? चलो अब पीछे वाले दोस्त को आगे आने दो!"
            else -> "क्षमा करें, चेहरा पहचाना नहीं जा सका, कृपया कैमरे के सामने सीधे देखें।"
        }
        speakHindi(message, boostFactor = 1.35f)
    }

    fun speakHindi(text: String, boostFactor: Float = 1.35f) {
        if (!isVoiceEnabled || text.isBlank()) return
        voiceExecutor.execute {
            speakHindiInternalBlocking(text, boostFactor)
        }
    }

    // केवल शुद्ध क्लाउड AI आवाज़ + लोकल मेमोरी कैश (कोई सिस्टम/फ़ोन ऑफ़लाइन TTS बिल्कुल नहीं!)
    private fun speakHindiInternalBlocking(text: String, boostFactor: Float) {
        scheduleSpeakingUnlock(10000L)

        val chosenVoice = aiVoiceProfiles[Random.nextInt(aiVoiceProfiles.size)]
        val otherVoice = aiVoiceProfiles.first { it.name != chosenVoice.name }

        val mp3CacheFile = File(voiceCacheDir, "${md5Hash("eleven_mp3_$text")}.mp3")
        val unifiedPcmFile = File(voiceCacheDir, "${md5Hash("gemini_pcm_$text")}.pcm")
        val file1 = File(voiceCacheDir, "${md5Hash("${chosenVoice.name}_$text")}.pcm")
        val file2 = File(voiceCacheDir, "${md5Hash("${otherVoice.name}_$text")}.pcm")

        // 1. पहले लोकल मेमोरी में सेव MP3 या PCM कैश चेक करें (एक बार बनी आवाज़ 0 सेकंड में बिना इंटरनेट/कोटा के चलेगी!)
        if (mp3CacheFile.exists() && mp3CacheFile.length() > 1200) {
            AiDiagnosticLogger.recordEvent(appCtx, "VOICE", "4. AI मानवीय आवाज़ (TTS)", "✅ कैश ऑडियो (ElevenLabs)", "लोकल मेमोरी में सेव मानवीय आवाज़ तुरंत चलाई गई")
            if (playMp3FileBlocking(mp3CacheFile)) return
        }

        for (pcmFile in listOf(unifiedPcmFile, file1, file2)) {
            if (pcmFile.exists() && pcmFile.length() > 3600) {
                val bytes = try { pcmFile.readBytes() } catch (e: Exception) { ByteArray(0) }
                if (isValidPcmAudioBytes(bytes)) {
                    AiDiagnosticLogger.recordEvent(appCtx, "VOICE", "4. AI मानवीय आवाज़ (TTS)", "✅ कैश ऑडियो (PCM)", "लोकल मेमोरी में सेव मानवीय आवाज़ तुरंत चलाई गई")
                    if (playRawPcm24kHzBlocking(bytes, boostFactor)) return
                } else {
                    try { pcmFile.delete() } catch (e: Exception) {}
                }
            }
        }

        val geminiPool = getGeminiKeyPool()
        val elevenPool = getElevenLabsKeyPool()

        if (geminiPool.isEmpty() && elevenPool.isEmpty()) {
            AiDiagnosticLogger.recordEvent(
                appCtx,
                "VOICE",
                "4. AI मानवीय आवाज़ (TTS)",
                "❌ कोई TTS Key नहीं",
                "कृपया Google AI Studio या ElevenLabs Key डालें"
            )
            isCurrentlySpeaking = false
            return
        }

        // 2. आपके 10+ Google AI Studio (Gemini TTS) Keys को सभी उपलब्ध Keys पर बारी-बारी से चलाएं (404 या 429 आने पर रुके बिना अगली Key आज़माएं!)
        if (geminiPool.isNotEmpty()) {
            val activeGemini = geminiPool.filter { isKeyAvailableNow(it) }.ifEmpty { geminiPool }
            val poolSize = activeGemini.size
            for (attempt in 0 until poolSize) {
                val idx = (geminiKeyIndex.getAndIncrement() % poolSize).let { if (it < 0) it + poolSize else it }
                val candidateKey = activeGemini[idx]
                val (pcmBytes, errCode, errReason) = fetchGoogleAiStudioTtsPcm(text, chosenVoice, candidateKey)
                if (pcmBytes != null && isValidPcmAudioBytes(pcmBytes)) {
                    writeFileAtomically(file1, pcmBytes)
                    writeFileAtomically(unifiedPcmFile, pcmBytes)
                    AiDiagnosticLogger.recordEvent(appCtx, "VOICE", "4. AI मानवीय आवाज़ (TTS)", "✅ सफल (Gemini TTS)", "Google AI Studio आवाज़ बनी और लोकल मेमोरी में सेव हो गई")
                    if (playRawPcm24kHzBlocking(pcmBytes, boostFactor)) return
                } else {
                    AiDiagnosticLogger.recordEvent(appCtx, "VOICE", "4. AI मानवीय आवाज़ (TTS)", "⚠️ Gemini Key रोटेशन", errReason)
                    if (errCode == -1) break // इंटरनेट बंद है
                    if (errCode == 400 || errCode == 403 || errCode == 404 || errCode == 429) {
                        markKeyCooldown(candidateKey, 5 * 60 * 1000L)
                        continue // अगली Google Key पर जाएं
                    }
                }
            }
        }

        // 3. फिर ElevenLabs Keys आज़माएं (बिना 12 बार लूप किए, 1 बार में तेज़ चेक)
        if (elevenPool.isNotEmpty()) {
            val activeEleven = elevenPool.filter { isKeyAvailableNow(it) }
            val poolSize = activeEleven.size
            for (attempt in 0 until poolSize) {
                val idx = (elevenKeyIndex.getAndIncrement() % poolSize).let { if (it < 0) it + poolSize else it }
                val candidateKey = activeEleven[idx]
                val (mp3Bytes, errCode, errReason) = fetchElevenLabsMp3WithFallback(text, chosenVoice.elevenVoiceIds, candidateKey)
                if (mp3Bytes != null && mp3Bytes.size > 1200) {
                    writeFileAtomically(mp3CacheFile, mp3Bytes)
                    AiDiagnosticLogger.recordEvent(appCtx, "VOICE", "4. AI मानवीय आवाज़ (TTS)", "✅ सफल (ElevenLabs)", "ElevenLabs मानवीय आवाज़ बनी और लोकल मेमोरी में सेव हो गई")
                    if (playMp3FileBlocking(mp3CacheFile)) return
                } else {
                    AiDiagnosticLogger.recordEvent(appCtx, "VOICE", "4. AI मानवीय आवाज़ (TTS)", "❌ ElevenLabs विफल", errReason)
                    if (errCode == -1) break
                    if (errCode == 401 || errCode == 402 || errCode == 403 || errCode == 429) {
                        markKeyCooldown(candidateKey, 10 * 60 * 1000L)
                        continue
                    }
                }
            }
        }

        isCurrentlySpeaking = false
    }

    private fun fetchGoogleAiStudioTtsPcm(
        hindiText: String,
        profile: AiVoiceProfile,
        apiKey: String
    ): Triple<ByteArray?, Int, String> {
        val cleanKey = apiKey.trim()
        val models = mutableListOf(
            "gemini-2.5-flash-preview-tts",
            "gemini-2.0-flash-exp",
            "gemini-2.5-pro-preview-tts"
        )
        discoveredModelForKey[cleanKey]?.let { bestModel ->
            models.remove(bestModel)
            models.add(0, bestModel)
        }

        var lastHttpCode = 0
        var lastReason = "अज्ञात कारण"

        for (modelName in models) {
            try {
                val endpoint = "https://generativelanguage.googleapis.com/v1beta/models/$modelName:generateContent?key=$cleanKey"
                val conn = (URL(endpoint).openConnection() as HttpURLConnection).apply {
                    requestMethod = "POST"
                    connectTimeout = 6500
                    readTimeout = 10000
                    doOutput = true
                    setRequestProperty("Content-Type", "application/json; charset=UTF-8")
                }

                val payloadJson = JSONObject().apply {
                    put("contents", JSONArray().put(
                        JSONObject().put("parts", JSONArray().put(
                            JSONObject().put("text", "${profile.stylePrompt} $hindiText")
                        ))
                    ))
                    put("generationConfig", JSONObject().apply {
                        put("responseModalities", JSONArray().put("AUDIO"))
                        put("speechConfig", JSONObject().apply {
                            put("voiceConfig", JSONObject().apply {
                                put("prebuiltVoiceConfig", JSONObject().apply {
                                    put("voiceName", profile.name)
                                })
                            })
                        })
                    })
                }

                conn.outputStream.use { os ->
                    os.write(payloadJson.toString().toByteArray(Charsets.UTF_8))
                }

                lastHttpCode = conn.responseCode
                if (lastHttpCode == 200) {
                    val responseStr = conn.inputStream.bufferedReader(Charsets.UTF_8).use { it.readText() }
                    val root = JSONObject(responseStr)
                    val parts = root.optJSONArray("candidates")
                        ?.optJSONObject(0)
                        ?.optJSONObject("content")
                        ?.optJSONArray("parts")
                    if (parts != null) {
                        for (i in 0 until parts.length()) {
                            val inlineObj = parts.optJSONObject(i)?.optJSONObject("inlineData")
                                ?: parts.optJSONObject(i)?.optJSONObject("inline_data")
                            val b64Audio = inlineObj?.optString("data", "") ?: ""
                            if (b64Audio.length > 200) {
                                discoveredModelForKey[cleanKey] = modelName
                                return Triple(Base64.decode(b64Audio, Base64.DEFAULT), 200, "OK")
                            }
                        }
                    }
                    lastReason = "सर्वर ने खाली ऑडियो भेजा"
                } else {
                    lastReason = when (lastHttpCode) {
                        400 -> "Gemini Key अमान्य है (HTTP 400)"
                        403 -> "Gemini Key पर अनुमति नहीं है (HTTP 403)"
                        404 -> "इस Key पर मॉडल ($modelName) उपलब्ध नहीं है (HTTP 404)"
                        429 -> "इस Key का कोटा पूरा हो गया (HTTP 429 — अगली Key पर स्विच)"
                        else -> "Gemini सर्वर त्रुटि (HTTP $lastHttpCode)"
                    }
                    if (lastHttpCode == 400 || lastHttpCode == 403 || lastHttpCode == 429) {
                        break // इस Key पर अन्य मॉडल न आज़माएं, सीधे अगली Key पर जाएं
                    }
                }
            } catch (e: UnknownHostException) {
                return Triple(null, -1, "मोबाइल का इंटरनेट बंद है (No Internet)")
            } catch (e: SocketTimeoutException) {
                lastReason = "इंटरनेट धीमा है (Connection Timeout)"
                break
            } catch (e: Exception) {
                lastReason = "नेटवर्क त्रुटि: ${e.javaClass.simpleName}"
                break
            }
        }
        return Triple(null, lastHttpCode, lastReason)
    }

    // ElevenLabs पर अनंत 12x लूप हटाकर केवल 1 सटीक प्रयास (ताकि unusual_activity या 401 आने पर 1 सेकंड भी बर्बाद न हो!)
    private fun fetchElevenLabsMp3WithFallback(
        hindiText: String,
        voiceIds: List<String>,
        elevenApiKey: String
    ): Triple<ByteArray?, Int, String> {
        val cleanKey = elevenApiKey.trim().replace("\"", "").replace("'", "")
        val primaryVoiceId = voiceIds.firstOrNull() ?: "21m00Tcm4TlvDq8ikWAM"
        val models = listOf("eleven_multilingual_v2", "eleven_flash_v2_5")
        var lastCode = 0
        var lastReason = "अज्ञात त्रुटि"

        for (modelId in models) {
            try {
                val endpoint = "https://api.elevenlabs.io/v1/text-to-speech/$primaryVoiceId"
                val conn = (URL(endpoint).openConnection() as HttpURLConnection).apply {
                    requestMethod = "POST"
                    connectTimeout = 6500
                    readTimeout = 10000
                    doOutput = true
                    setRequestProperty("xi-api-key", cleanKey)
                    setRequestProperty("Content-Type", "application/json; charset=UTF-8")
                    setRequestProperty("Accept", "audio/mpeg")
                    setRequestProperty("User-Agent", "Mozilla/5.0 (Linux; Android 14) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Mobile Safari/537.36")
                }

                val payload = JSONObject().apply {
                    put("text", hindiText)
                    put("model_id", modelId)
                    put("voice_settings", JSONObject().apply {
                        put("stability", 0.50)
                        put("similarity_boost", 0.78)
                    })
                }

                conn.outputStream.use { os ->
                    os.write(payload.toString().toByteArray(Charsets.UTF_8))
                }

                lastCode = conn.responseCode
                if (lastCode == 200) {
                    val bytes = conn.inputStream.readBytes()
                    if (bytes.size > 1200) {
                        return Triple(bytes, 200, "OK")
                    }
                    lastReason = "ElevenLabs से बहुत छोटी ऑडियो प्राप्त हुई"
                } else {
                    val errBody = try { conn.errorStream?.bufferedReader()?.readText() ?: "" } catch (e: Exception) { "" }
                    lastReason = AiDiagnosticLogger.parseElevenLabsErrorDetail(errBody, lastCode)
                    // 401 (unusual_activity / missing_permissions), 402, 403, 429 आने पर दोबारा हिट न करें, तुरंत बाहर निकलें!
                    if (lastCode == 401 || lastCode == 402 || lastCode == 403 || lastCode == 429) {
                        return Triple(null, lastCode, lastReason)
                    }
                }
            } catch (e: UnknownHostException) {
                return Triple(null, -1, "मोबाइल का इंटरनेट बंद है (No Internet)")
            } catch (e: SocketTimeoutException) {
                return Triple(null, -2, "इंटरनेट बहुत धीमा है (ElevenLabs Timeout)")
            } catch (e: Exception) {
                return Triple(null, -3, "कनेक्शन त्रुटि: ${e.javaClass.simpleName}")
            }
        }
        return Triple(null, lastCode, lastReason)
    }

    private fun playMp3FileBlocking(mp3File: File): Boolean {
        val doneLatch = CountDownLatch(1)
        var startedOk = false
        var playDurationMs = 4500L
        return try {
            stopCurrentAudio()
            val mp = MediaPlayer()
            mediaPlayer = mp
            mp.setAudioAttributes(
                AudioAttributes.Builder()
                    .setUsage(AudioAttributes.USAGE_MEDIA)
                    .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH)
                    .build()
            )
            mp.setDataSource(mp3File.absolutePath)
            mp.prepare()
            mp.setVolume(1.0f, 1.0f)
            try {
                loudnessEnhancer = LoudnessEnhancer(mp.audioSessionId).apply {
                    setTargetGain(650)
                    enabled = true
                }
            } catch (e: Exception) {}

            playDurationMs = mp.duration.toLong().coerceIn(1500L, 20000L)
            scheduleSpeakingUnlock(playDurationMs + 300L)

            mp.setOnCompletionListener {
                doneLatch.countDown()
            }
            mp.setOnErrorListener { _, _, _ ->
                doneLatch.countDown()
                true
            }
            mp.start()
            startedOk = true
            doneLatch.await(playDurationMs + 800L, TimeUnit.MILLISECONDS)
            isCurrentlySpeaking = false
            startedOk
        } catch (e: Exception) {
            isCurrentlySpeaking = false
            false
        }
    }

    private fun amplifyPcmSamples(rawPcm: ByteArray, gain: Float): ByteArray {
        if (gain <= 1.0f || rawPcm.size < 2) return rawPcm
        val safeLen = rawPcm.size - (rawPcm.size % 2)
        val out = ByteArray(safeLen)
        val inBuf = ByteBuffer.wrap(rawPcm, 0, safeLen).order(ByteOrder.LITTLE_ENDIAN)
        val outBuf = ByteBuffer.wrap(out).order(ByteOrder.LITTLE_ENDIAN)

        while (inBuf.remaining() >= 2) {
            val sample = inBuf.short.toInt()
            val boosted = (sample * gain).toInt().coerceIn(Short.MIN_VALUE.toInt(), Short.MAX_VALUE.toInt())
            outBuf.putShort(boosted.toShort())
        }
        return out
    }

    // AudioTrack.MODE_STREAM द्वारा स्ट्रीमिंग प्लेबैक: जिससे छोटा या 15 सेकंड लंबा VIP ऑडियो हर फोन पर 100% साफ़ बजे और कभी मेमोरी लिमिट से मूक न हो!
    private fun playRawPcm24kHzBlocking(pcmData: ByteArray, boostFactor: Float): Boolean {
        return try {
            stopCurrentAudio()
            val boostedData = amplifyPcmSamples(pcmData, boostFactor)
            if (boostedData.size < 1000) {
                isCurrentlySpeaking = false
                return false
            }

            val sampleRate = 24000
            val minBufSize = AudioTrack.getMinBufferSize(
                sampleRate,
                AudioFormat.CHANNEL_OUT_MONO,
                AudioFormat.ENCODING_PCM_16BIT
            )
            val streamBufSize = max(minBufSize * 2, 16384)

            val track = AudioTrack.Builder()
                .setAudioAttributes(
                    AudioAttributes.Builder()
                        .setUsage(AudioAttributes.USAGE_MEDIA)
                        .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH)
                        .build()
                )
                .setAudioFormat(
                    AudioFormat.Builder()
                        .setEncoding(AudioFormat.ENCODING_PCM_16BIT)
                        .setSampleRate(sampleRate)
                        .setChannelMask(AudioFormat.CHANNEL_OUT_MONO)
                        .build()
                )
                .setBufferSizeInBytes(streamBufSize)
                .setTransferMode(AudioTrack.MODE_STREAM)
                .build()

            track.setVolume(AudioTrack.getMaxVolume())

            try {
                loudnessEnhancer = LoudnessEnhancer(track.audioSessionId).apply {
                    setTargetGain(600)
                    enabled = true
                }
            } catch (e: Exception) {}

            val exactDurationMs = ((boostedData.size * 1000L) / (sampleRate * 2)).coerceIn(1000L, 22000L)
            scheduleSpeakingUnlock(exactDurationMs + 350L)

            currentAudioTrack = track
            track.play()

            var offset = 0
            val chunkSize = 8192
            while (offset < boostedData.size) {
                val toWrite = min(chunkSize, boostedData.size - offset)
                val written = track.write(boostedData, offset, toWrite)
                if (written <= 0) break
                offset += written
            }

            Thread.sleep(220L)
            try {
                track.stop()
                track.release()
            } catch (e: Exception) {}
            if (currentAudioTrack === track) {
                currentAudioTrack = null
            }
            isCurrentlySpeaking = false
            true
        } catch (e: Exception) {
            isCurrentlySpeaking = false
            false
        }
    }

    private fun stopCurrentAudio() {
        try {
            loudnessEnhancer?.release()
            loudnessEnhancer = null
        } catch (e: Exception) {}

        try {
            currentAudioTrack?.let {
                if (it.playState == AudioTrack.PLAYSTATE_PLAYING) it.stop()
                it.release()
            }
            currentAudioTrack = null
        } catch (e: Exception) {}

        try {
            mediaPlayer?.let {
                if (it.isPlaying) it.stop()
                it.release()
            }
            mediaPlayer = null
        } catch (e: Exception) {}
    }

    private fun md5Hash(input: String): String {
        val bytes = MessageDigest.getInstance("MD5").digest(input.toByteArray())
        return bytes.joinToString("") { "%02x".format(it) }
    }

    fun shutdown() {
        isCurrentlySpeaking = false
        stopCurrentAudio()
    }
}
