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
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicInteger
import kotlin.math.PI
import kotlin.math.exp
import kotlin.math.sin
import kotlin.random.Random

object AiDiagnosticLogger {
    private const val PREF_DIAG = "SarvodayaAiDiagPrefs"
    private const val KEY_LOGS = "diag_logs_json"
    private const val KEY_STATUS_MAP = "diag_status_map_json"

    private fun nowTime(): String {
        return SimpleDateFormat("dd.MM hh:mm:ss a", Locale.US).format(Date())
    }

    fun parseElevenLabsErrorDetail(httpCode: Int, rawErrBody: String): String {
        return try {
            val detailObj = JSONObject(rawErrBody).optJSONObject("detail")
            val status = detailObj?.optString("status", "") ?: ""
            val msg = detailObj?.optString("message", "") ?: ""
            when {
                status.contains("unusual_activity", ignoreCase = true) || msg.contains("Unusual activity", ignoreCase = true) ->
                    "ElevenLabs ने आपके मोबाइल नेटवर्क/IP पर Free Tier रोका है (unusual_activity_detected)"
                status.contains("missing_permissions", ignoreCase = true) || msg.contains("missing the permission", ignoreCase = true) ->
                    "इस Key में 'Text to Speech' की अनुमति बंद है (ElevenLabs में नई Full-Access Key बनाएं)"
                status.contains("quota_exceeded", ignoreCase = true) ->
                    "ElevenLabs का मासिक करैक्टर कोटा समाप्त हो गया है"
                msg.isNotBlank() ->
                    "HTTP $httpCode ($status): ${msg.take(75)}"
                else ->
                    "ElevenLabs अस्वीकृत (HTTP $httpCode)"
            }
        } catch (e: Exception) {
            if (httpCode == 401) "ElevenLabs HTTP 401 (Key में Text-to-Speech अनुमति बंद है या नेटवर्क IP ब्लॉक है)"
            else "ElevenLabs सर्वर कोड: $httpCode"
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
        for (k in gKeys) {
            val masked = if (k.length > 10) "${k.take(6)}••••••••${k.takeLast(4)}" else k
            var status = "❌ अमान्य"
            var detail = "कोई रिस्पॉन्स नहीं"
            try {
                val url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key=$k"
                val conn = (URL(url).openConnection() as HttpURLConnection).apply {
                    requestMethod = "POST"
                    connectTimeout = 6000
                    readTimeout = 7000
                    doOutput = true
                    setRequestProperty("Content-Type", "application/json; charset=UTF-8")
                }
                val body = JSONObject().put("contents", JSONArray().put(JSONObject().put("parts", JSONArray().put(JSONObject().put("text", "OK")))))
                conn.outputStream.use { it.write(body.toString().toByteArray(Charsets.UTF_8)) }
                val code = conn.responseCode
                if (code == 200) {
                    status = "✅ सक्रिय (Active)"
                    detail = "Gemini 2.5 Flash तैयार है"
                    recordEvent(appCtx, "SLOGAN", "5. दैनिक AI स्लोगन", "✅ सक्रिय", "Gemini Key लाइव जांच में सफल")
                } else {
                    status = "❌ त्रुटि (HTTP $code)"
                    detail = when (code) {
                        400 -> "Key गलत है (Invalid Key)"
                        403 -> "अनुमति नहीं है (Forbidden)"
                        429 -> "दैनिक कोटा समाप्त (HTTP 429)"
                        else -> "सर्वर कोड $code"
                    }
                }
            } catch (e: UnknownHostException) {
                status = "❌ इंटरनेट बंद"
                detail = "डेटा या वाई-फाई चालू करें"
            } catch (e: SocketTimeoutException) {
                status = "⚠️ नेटवर्क धीमा"
                detail = "कनेक्शन टाइमआउट"
            } catch (e: Exception) {
                status = "❌ त्रुटि"
                detail = e.javaClass.simpleName
            }
            geminiResults.put(JSONObject().apply {
                put("maskedKey", masked)
                put("status", status)
                put("detail", detail)
            })
        }

        val elevenResults = JSONArray()
        for (k in eKeys) {
            val masked = if (k.length > 10) "${k.take(6)}••••••••${k.takeLast(4)}" else k
            var status = "❌ अमान्य"
            var detail = ""
            try {
                val subUrl = "https://api.elevenlabs.io/v1/user/subscription"
                val subConn = (URL(subUrl).openConnection() as HttpURLConnection).apply {
                    requestMethod = "GET"
                    connectTimeout = 6000
                    readTimeout = 7000
                    setRequestProperty("xi-api-key", k)
                }
                val subCode = subConn.responseCode
                var quotaText = ""
                if (subCode == 200) {
                    val resp = subConn.inputStream.bufferedReader(Charsets.UTF_8).use { it.readText() }
                    val sub = JSONObject(resp)
                    val tier = sub.optString("tier", "free").uppercase()
                    val used = sub.optInt("character_count", 0)
                    val limit = sub.optInt("character_limit", 10000)
                    val rem = (limit - used).coerceAtLeast(0)
                    val nextUnix = sub.optLong("next_character_count_reset_unix", 0L)
                    val resetDate = if (nextUnix > 0L) {
                        SimpleDateFormat("dd MMM yyyy", Locale.US).format(Date(nextUnix * 1000L))
                    } else "-"
                    quotaText = "बचे अक्षर: $rem / $limit (Tier: $tier) | Reset: $resetDate"
                }

                // अब वास्तविक Text-to-Speech (TTS) परमिशन व ऑडियो जनरेशन की लाइव जांच करें
                val ttsTest = vm.testElevenLabsTtsLive(k)
                if (ttsTest.first) {
                    status = "✅ वॉइस चालू है (TTS Active)"
                    detail = if (quotaText.isNotEmpty()) quotaText else "ElevenLabs ऑडियो जनरेशन सफल"
                    recordEvent(appCtx, "VOICE", "4. AI मानवीय आवाज़ (TTS)", "✅ ElevenLabs सक्रिय", detail)
                } else {
                    status = "⚠️ TTS अवरुद्ध (${ttsTest.second})"
                    detail = if (quotaText.isNotEmpty()) "$quotaText\nकारण: ${ttsTest.third}" else ttsTest.third
                    recordEvent(appCtx, "VOICE", "4. AI मानवीय आवाज़ (TTS)", "❌ ElevenLabs विफल", ttsTest.third)
                }
            } catch (e: UnknownHostException) {
                status = "❌ इंटरनेट बंद"
                detail = "मोबाइल डेटा या वाई-फाई चालू करें"
            } catch (e: SocketTimeoutException) {
                status = "⚠️ नेटवर्क धीमा"
                detail = "कनेक्शन टाइमआउट"
            } catch (e: Exception) {
                status = "❌ त्रुटि"
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
}

class RealisticVoiceManager(private val context: Context) {

    private val appCtx = context.applicationContext
    private val prefs = appCtx.getSharedPreferences("SarvodayaVoicePrefs", Context.MODE_PRIVATE)
    private val mainHandler = Handler(Looper.getMainLooper())

    private val geminiKeyIndex = AtomicInteger(0)
    private val elevenKeyIndex = AtomicInteger(0)

    @Volatile
    private var cachedDynamicElevenVoiceId: String? = null

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

        cachedDynamicElevenVoiceId = null

        AiDiagnosticLogger.recordEvent(
            appCtx,
            "VOICE",
            "4. AI मानवीय आवाज़ (TTS)",
            "🔄 Keys सेव",
            "Gemini: ${if (gClean.isEmpty()) 0 else gClean.lines().size}, ElevenLabs: ${if (eClean.isEmpty()) 0 else eClean.lines().size}"
        )

        rollingCacheExecutor.execute {
            try {
                voiceCacheDir.listFiles()?.forEach { file ->
                    if (file.name.startsWith("welcome_slot_")) {
                        try { file.delete() } catch (e: Exception) {}
                    }
                }
            } catch (e: Exception) {}
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

    private fun getGeminiKeyPool(): List<String> {
        val raw = getGeminiKeysString()
        return raw.split(",", "\n", ";", " ")
            .map { it.trim() }
            .filter { it.length > 15 && !it.startsWith("sk_", ignoreCase = true) }
    }

    private fun getElevenLabsKeyPool(): List<String> {
        val raw = getElevenLabsKeysString()
        return raw.split(",", "\n", ";", " ")
            .map { it.trim() }
            .filter { it.length > 15 && !it.startsWith("AIza") && !it.startsWith("AQ.") }
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
                "21m00Tcm4TlvDq8ikWAM", // Rachel (Default Premade Free Voice)
                "pNInz6obpgDQGcFmaJgB", // Adam (Default Premade Free Voice)
                "EXAVITQu4vr4xnSDxMaL"  // Sarah
            ),
            stylePrompt = "Speak like an energetic, proud, warm Indian Hindi radio RJ host with a natural smile and lively pace:"
        ),
        AiVoiceProfile(
            name = "Autonoe",
            elevenVoiceIds = listOf(
                "21m00Tcm4TlvDq8ikWAM", // Rachel
                "AZnzlk1XvdvUeBnXmlld", // Domi
                "pNInz6obpgDQGcFmaJgB"  // Adam
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

    private val voiceCacheDir: File by lazy {
        File(appCtx.getExternalFilesDir(null) ?: appCtx.filesDir, "ai_studio_voice_bank").apply {
            if (!exists()) mkdirs()
        }
    }

    init {
        maintainTwoHourRollingWelcomeCache()
    }

    // ElevenLabs अकाउंट से उपलब्ध पहली वैध Voice ID अपने-आप प्राप्त करना
    private fun fetchAvailableAccountVoiceId(elevenKey: String): String? {
        cachedDynamicElevenVoiceId?.let { return it }
        return try {
            val conn = (URL("https://api.elevenlabs.io/v1/voices").openConnection() as HttpURLConnection).apply {
                requestMethod = "GET"
                connectTimeout = 6000
                readTimeout = 6000
                setRequestProperty("xi-api-key", elevenKey.trim())
            }
            if (conn.responseCode == 200) {
                val resp = conn.inputStream.bufferedReader(Charsets.UTF_8).use { it.readText() }
                val voices = JSONObject(resp).optJSONArray("voices")
                if (voices != null && voices.length() > 0) {
                    val vId = voices.optJSONObject(0)?.optString("voice_id", "") ?: ""
                    if (vId.isNotBlank()) {
                        cachedDynamicElevenVoiceId = vId
                        return vId
                    }
                }
            }
            null
        } catch (e: Exception) {
            null
        }
    }

    // लाइव चेक बटन के लिए वास्तविक TTS टेस्ट
    fun testElevenLabsTtsLive(elevenKey: String): Triple<Boolean, String, String> {
        val (bytes, code, reason) = fetchElevenLabsMp3WithFallback(
            "नमस्ते",
            listOf("21m00Tcm4TlvDq8ikWAM", "pNInz6obpgDQGcFmaJgB"),
            elevenKey
        )
        return if (bytes != null && bytes.size > 500) {
            Triple(true, "HTTP 200", "OK")
        } else {
            Triple(false, "HTTP $code", reason)
        }
    }

    private fun getCurrentTwoHourSlot(): Long {
        return System.currentTimeMillis() / (2L * 60L * 60L * 1000L)
    }

    private fun maintainTwoHourRollingWelcomeCache() {
        rollingCacheExecutor.execute {
            try {
                val currentSlot = getCurrentTwoHourSlot()
                val nextSlot = currentSlot + 1L

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

                if (!isSlotAudioReady(currentSlot)) {
                    generateAndCacheSlotAudio(currentSlot, isNextSlot = false)
                }

                if (!isSlotAudioReady(nextSlot)) {
                    generateAndCacheSlotAudio(nextSlot, isNextSlot = true)
                }
            } catch (e: Exception) {}
        }
    }

    private fun isSlotAudioReady(slotId: Long): Boolean {
        val pcm = File(voiceCacheDir, "welcome_slot_$slotId.pcm")
        val mp3 = File(voiceCacheDir, "welcome_slot_$slotId.mp3")
        return (pcm.exists() && pcm.length() > 2000) || (mp3.exists() && mp3.length() > 1000)
    }

    private fun generateAndCacheSlotAudio(slotId: Long, isNextSlot: Boolean): Boolean {
        val welcomeText = generateFreshAiWelcomeLineForSlot(slotId, isNextSlot)
        val profile = aiVoiceProfiles[(slotId % aiVoiceProfiles.size).toInt().let { if (it < 0) -it else it }]
        val targetPcm = File(voiceCacheDir, "welcome_slot_$slotId.pcm")
        val targetMp3 = File(voiceCacheDir, "welcome_slot_$slotId.mp3")

        val geminiPool = getGeminiKeyPool()
        for (attempt in geminiPool.indices) {
            val idx = (geminiKeyIndex.getAndIncrement() % geminiPool.size).let { if (it < 0) it + geminiPool.size else it }
            val (pcmBytes, code, errReason) = fetchGoogleAiStudioTtsPcm(welcomeText, profile, geminiPool[idx])
            if (pcmBytes != null && pcmBytes.size > 2000) {
                writeFileAtomically(targetPcm, pcmBytes)
                AiDiagnosticLogger.recordEvent(appCtx, "VOICE", "4. AI मानवीय आवाज़ (TTS)", "✅ Gemini TTS सक्रिय", "स्वागत ऑडियो तैयार")
                return true
            } else {
                AiDiagnosticLogger.recordEvent(appCtx, "VOICE", "4. AI मानवीय आवाज़ (TTS)", "⚠️ Gemini TTS विफल", errReason)
            }
            if (code != 429 && code != 400 && code != 403) break
        }

        val elevenPool = getElevenLabsKeyPool()
        for (attempt in elevenPool.indices) {
            val idx = (elevenKeyIndex.getAndIncrement() % elevenPool.size).let { if (it < 0) it + elevenPool.size else it }
            val (mp3Bytes, code, errReason) = fetchElevenLabsMp3WithFallback(welcomeText, profile.elevenVoiceIds, elevenPool[idx])
            if (mp3Bytes != null && mp3Bytes.size > 1000) {
                writeFileAtomically(targetMp3, mp3Bytes)
                AiDiagnosticLogger.recordEvent(appCtx, "VOICE", "4. AI मानवीय आवाज़ (TTS)", "✅ ElevenLabs सक्रिय", "स्वागत ऑडियो तैयार")
                return true
            } else {
                AiDiagnosticLogger.recordEvent(appCtx, "VOICE", "4. AI मानवीय आवाज़ (TTS)", "❌ ElevenLabs विफल", errReason)
            }
            if (code != 429 && code != 401 && code != 403) break
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
            "प्रेरणादायक और जोशपूर्ण",
            "हाई-टेक डिजिटल स्मार्ट कैंपस",
            "बच्चों और शिक्षकों से आत्मीय संवाद",
            "अनुशासन, मुस्कान और गर्व से भरा"
        )
        val chosenStyle = styles[(slotId % styles.size).toInt().let { if (it < 0) -it else it }]

        val geminiPool = getGeminiKeyPool()
        if (geminiPool.isNotEmpty()) {
            try {
                val apiKey = geminiPool[0]
                val endpoint = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key=$apiKey"
                val conn = (URL(endpoint).openConnection() as HttpURLConnection).apply {
                    requestMethod = "POST"
                    connectTimeout = 6000
                    readTimeout = 8000
                    doOutput = true
                    setRequestProperty("Content-Type", "application/json; charset=UTF-8")
                }

                val prompt = """
                    विद्यालय का नाम: "सर्वोदय मध्य विद्यालय, सोहसराय"।
                    अभिवादन शब्द: "$greetingWord"।
                    अंदाज़: $chosenStyle।
                    नियम: केवल 18 से 24 शब्दों में एक नया, आकर्षक रेडियो एंकर जैसा हिंदी स्वागत वाक्य लिखें जो छात्रों को कैमरे के सामने मुस्कुराकर स्मार्ट उपस्थिति दर्ज कराने के लिए प्रेरित करे। केवल शुद्ध वाक्य लौटाएं। (Slot: $slotId)
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
                        AiDiagnosticLogger.recordEvent(appCtx, "SLOGAN", "5. दैनिक AI स्लोगन", "✅ सफल (Gemini)", "\"${text.take(45)}...\"")
                        return text
                    }
                } else {
                    AiDiagnosticLogger.recordEvent(appCtx, "SLOGAN", "5. दैनिक AI स्लोगन", "⚠️ डिफ़ॉल्ट स्लोगन", "Gemini HTTP $code")
                }
            } catch (e: Exception) {
                AiDiagnosticLogger.recordEvent(appCtx, "SLOGAN", "5. दैनिक AI स्लोगन", "ℹ️ इन-बिल्ट स्लोगन", "ऑफलाइन स्लोगन सक्रिय")
            }
        } else {
            AiDiagnosticLogger.recordEvent(appCtx, "SLOGAN", "5. दैनिक AI स्लोगन", "ℹ️ इन-बिल्ट स्लोगन", "इन-बिल्ट हिंदी स्लोगन तैयार")
        }

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
            if (tempFile.exists() && tempFile.length() > 800) {
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
            val durationMs = 440
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
                .setBufferSizeInBytes(kotlin.math.max(minBuf, pcmBytes.size))
                .setTransferMode(AudioTrack.MODE_STATIC)
                .build()

            track.write(pcmBytes, 0, pcmBytes.size)
            track.play()
            Thread.sleep(durationMs.toLong() + 40L)
            track.stop()
            track.release()
        } catch (e: Exception) {}
    }

    fun speakStartupWelcome() {
        if (!isVoiceEnabled) return
        voiceExecutor.execute {
            val currentSlot = getCurrentTwoHourSlot()
            val slotPcm = File(voiceCacheDir, "welcome_slot_$currentSlot.pcm")
            val slotMp3 = File(voiceCacheDir, "welcome_slot_$currentSlot.mp3")

            if (slotPcm.exists() && slotPcm.length() > 2000) {
                playSoftStartupChimeBlocking()
                playRawPcm24kHz(slotPcm.readBytes(), 1.55f)
                maintainTwoHourRollingWelcomeCache()
                return@execute
            }
            if (slotMp3.exists() && slotMp3.length() > 1000) {
                playSoftStartupChimeBlocking()
                playMp3File(slotMp3)
                maintainTwoHourRollingWelcomeCache()
                return@execute
            }

            if (generateAndCacheSlotAudio(currentSlot, isNextSlot = false)) {
                if (slotPcm.exists() && slotPcm.length() > 2000) {
                    playSoftStartupChimeBlocking()
                    playRawPcm24kHz(slotPcm.readBytes(), 1.55f)
                } else if (slotMp3.exists() && slotMp3.length() > 1000) {
                    playSoftStartupChimeBlocking()
                    playMp3File(slotMp3)
                }
            }
            maintainTwoHourRollingWelcomeCache()
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
            speakHindiInternal(text, boostFactor)
        }
    }

    private fun speakHindiInternal(text: String, boostFactor: Float) {
        scheduleSpeakingUnlock(10000L)

        val chosenVoice = aiVoiceProfiles[Random.nextInt(aiVoiceProfiles.size)]
        val otherVoice = aiVoiceProfiles.first { it.name != chosenVoice.name }

        val file1 = File(voiceCacheDir, "${md5Hash("${chosenVoice.name}_$text")}.pcm")
        val file2 = File(voiceCacheDir, "${md5Hash("${otherVoice.name}_$text")}.pcm")
        val mp3CacheFile = File(voiceCacheDir, "${md5Hash("eleven_mp3_$text")}.mp3")

        if (file1.exists() && file1.length() > 2000) {
            AiDiagnosticLogger.recordEvent(appCtx, "VOICE", "4. AI मानवीय आवाज़ (TTS)", "✅ कैश ऑडियो (PCM)", "सेव की गई आवाज़ चलाई गई")
            if (playRawPcm24kHz(file1.readBytes(), boostFactor)) return
        }
        if (file2.exists() && file2.length() > 2000) {
            AiDiagnosticLogger.recordEvent(appCtx, "VOICE", "4. AI मानवीय आवाज़ (TTS)", "✅ कैश ऑडियो (PCM)", "सेव की गई आवाज़ चलाई गई")
            if (playRawPcm24kHz(file2.readBytes(), boostFactor)) return
        }
        if (mp3CacheFile.exists() && mp3CacheFile.length() > 1000) {
            AiDiagnosticLogger.recordEvent(appCtx, "VOICE", "4. AI मानवीय आवाज़ (TTS)", "✅ कैश ऑडियो (MP3)", "सेव की गई ElevenLabs आवाज़ चलाई गई")
            if (playMp3File(mp3CacheFile)) return
        }

        val geminiPool = getGeminiKeyPool()
        val elevenPool = getElevenLabsKeyPool()

        if (geminiPool.isEmpty() && elevenPool.isEmpty()) {
            AiDiagnosticLogger.recordEvent(
                appCtx,
                "VOICE",
                "4. AI मानवीय आवाज़ (TTS)",
                "❌ कोई Key नहीं",
                "कोई API Key सेव नहीं है"
            )
            isCurrentlySpeaking = false
            return
        }

        if (geminiPool.isNotEmpty()) {
            val poolSize = geminiPool.size
            for (attempt in 0 until poolSize) {
                val idx = (geminiKeyIndex.getAndIncrement() % poolSize).let { if (it < 0) it + poolSize else it }
                val candidateKey = geminiPool[idx]
                val (pcmBytes, errCode, errReason) = fetchGoogleAiStudioTtsPcm(text, chosenVoice, candidateKey)
                if (pcmBytes != null && pcmBytes.size > 2000) {
                    writeFileAtomically(file1, pcmBytes)
                    AiDiagnosticLogger.recordEvent(appCtx, "VOICE", "4. AI मानवीय आवाज़ (TTS)", "✅ सफल (Gemini TTS)", "आवाज़ सफलतापूर्वक चली")
                    if (playRawPcm24kHz(pcmBytes, boostFactor)) return
                } else {
                    AiDiagnosticLogger.recordEvent(appCtx, "VOICE", "4. AI मानवीय आवाज़ (TTS)", "⚠️ Gemini विफल", errReason)
                }
                if (errCode != 429 && errCode != 400 && errCode != 403) break
            }
        }

        if (elevenPool.isNotEmpty()) {
            val poolSize = elevenPool.size
            for (attempt in 0 until poolSize) {
                val idx = (elevenKeyIndex.getAndIncrement() % poolSize).let { if (it < 0) it + poolSize else it }
                val candidateKey = elevenPool[idx]
                val (mp3Bytes, errCode, errReason) = fetchElevenLabsMp3WithFallback(text, chosenVoice.elevenVoiceIds, candidateKey)
                if (mp3Bytes != null && mp3Bytes.size > 1000) {
                    writeFileAtomically(mp3CacheFile, mp3Bytes)
                    AiDiagnosticLogger.recordEvent(appCtx, "VOICE", "4. AI मानवीय आवाज़ (TTS)", "✅ सफल (ElevenLabs)", "आवाज़ सफलतापूर्वक चली")
                    if (playMp3File(mp3CacheFile)) return
                } else {
                    AiDiagnosticLogger.recordEvent(appCtx, "VOICE", "4. AI मानवीय आवाज़ (TTS)", "❌ ElevenLabs विफल", errReason)
                }
                if (errCode != 429 && errCode != 401 && errCode != 403) break
            }
        }

        isCurrentlySpeaking = false
    }

    private fun fetchGoogleAiStudioTtsPcm(
        hindiText: String,
        profile: AiVoiceProfile,
        apiKey: String
    ): Triple<ByteArray?, Int, String> {
        val models = listOf("gemini-2.5-flash-preview-tts", "gemini-2.5-pro-preview-tts")
        var lastHttpCode = 0
        var lastReason = "अज्ञात कारण"

        for (modelName in models) {
            try {
                val endpoint = "https://generativelanguage.googleapis.com/v1beta/models/$modelName:generateContent?key=$apiKey"
                val conn = (URL(endpoint).openConnection() as HttpURLConnection).apply {
                    requestMethod = "POST"
                    connectTimeout = 9000
                    readTimeout = 14000
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
                    val part = root.optJSONArray("candidates")
                        ?.optJSONObject(0)
                        ?.optJSONObject("content")
                        ?.optJSONArray("parts")
                        ?.optJSONObject(0)
                    val b64Audio = part?.optJSONObject("inlineData")?.optString("data", "") ?: ""
                    if (b64Audio.isNotEmpty()) {
                        return Triple(Base64.decode(b64Audio, Base64.DEFAULT), 200, "OK")
                    }
                    lastReason = "सर्वर ने खाली ऑडियो भेजा"
                } else {
                    lastReason = when (lastHttpCode) {
                        400 -> "Gemini Key अमान्य है (HTTP 400)"
                        403 -> "अनुमति नहीं है (HTTP 403)"
                        429 -> "Gemini TTS कोटा समाप्त (HTTP 429)"
                        else -> "Gemini सर्वर कोड $lastHttpCode"
                    }
                    if (lastHttpCode == 400 || lastHttpCode == 403 || lastHttpCode == 429) {
                        break
                    }
                }
            } catch (e: UnknownHostException) {
                lastReason = "इंटरनेट बंद है"
            } catch (e: SocketTimeoutException) {
                lastReason = "इंटरनेट धीमा है (Timeout)"
            } catch (e: Exception) {
                lastReason = "नेटवर्क त्रुटि: ${e.javaClass.simpleName}"
            }
        }
        return Triple(null, lastHttpCode, lastReason)
    }

    private fun fetchElevenLabsMp3WithFallback(
        hindiText: String,
        voiceIds: List<String>,
        elevenApiKey: String
    ): Triple<ByteArray?, Int, String> {
        val cleanKey = elevenApiKey.trim()
        val dynamicVoice = fetchAvailableAccountVoiceId(cleanKey)
        val combinedVoices = LinkedHashSet<String>().apply {
            if (!dynamicVoice.isNullOrBlank()) add(dynamicVoice)
            addAll(voiceIds)
        }.toList()

        val models = listOf("eleven_multilingual_v2", "eleven_flash_v2_5", "eleven_turbo_v2_5")
        var lastCode = 0
        var lastReason = "अज्ञात त्रुटि"

        for (vId in combinedVoices) {
            for (modelId in models) {
                try {
                    val endpoint = "https://api.elevenlabs.io/v1/text-to-speech/$vId"
                    val conn = (URL(endpoint).openConnection() as HttpURLConnection).apply {
                        requestMethod = "POST"
                        connectTimeout = 10000
                        readTimeout = 15000
                        doOutput = true
                        setRequestProperty("xi-api-key", cleanKey)
                        setRequestProperty("Content-Type", "application/json; charset=UTF-8")
                        setRequestProperty("Accept", "audio/mpeg")
                        setRequestProperty("User-Agent", "Mozilla/5.0 (Android 14; Mobile)")
                    }

                    val payload = JSONObject().apply {
                        put("text", hindiText)
                        put("model_id", modelId)
                        put("voice_settings", JSONObject().apply {
                            put("stability", 0.50)
                            put("similarity_boost", 0.75)
                        })
                    }

                    conn.outputStream.use { os ->
                        os.write(payload.toString().toByteArray(Charsets.UTF_8))
                    }

                    lastCode = conn.responseCode
                    if (lastCode == 200) {
                        val bytes = conn.inputStream.readBytes()
                        if (bytes.size > 500) {
                            return Triple(bytes, 200, "OK")
                        }
                        lastReason = "छोटी ऑडियो प्राप्त हुई"
                    } else {
                        val errBody = try { conn.errorStream?.bufferedReader()?.readText() ?: "" } catch (e: Exception) { "" }
                        lastReason = AiDiagnosticLogger.parseElevenLabsErrorDetail(lastCode, errBody)

                        // यदि Key में परमिशन नहीं है या IP पर Free Tier ब्लॉक है, तो बार-बार लूप न चलाएं
                        if (errBody.contains("missing_permissions", ignoreCase = true) ||
                            errBody.contains("unusual_activity", ignoreCase = true) ||
                            errBody.contains("invalid_api_key", ignoreCase = true)
                        ) {
                            return Triple(null, lastCode, lastReason)
                        }

                        if (lastCode == 402 || errBody.contains("voice", ignoreCase = true)) {
                            break
                        }
                    }
                } catch (e: UnknownHostException) {
                    return Triple(null, 0, "इंटरनेट बंद है (No Internet)")
                } catch (e: SocketTimeoutException) {
                    lastReason = "इंटरनेट धीमा है (ElevenLabs Timeout)"
                } catch (e: Exception) {
                    lastReason = "कनेक्शन त्रुटि: ${e.javaClass.simpleName}"
                }
            }
        }
        return Triple(null, lastCode, lastReason)
    }

    private fun playMp3File(mp3File: File): Boolean {
        return try {
            stopCurrentAudio()
            mediaPlayer = MediaPlayer().apply {
                setAudioAttributes(
                    AudioAttributes.Builder()
                        .setUsage(AudioAttributes.USAGE_MEDIA)
                        .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH)
                        .build()
                )
                setDataSource(mp3File.absolutePath)
                prepare()
                setVolume(1.0f, 1.0f)
                try {
                    loudnessEnhancer = LoudnessEnhancer(audioSessionId).apply {
                        setTargetGain(650)
                        enabled = true
                    }
                } catch (e: Exception) {}
                val dur = duration.toLong().coerceAtLeast(3200L)
                scheduleSpeakingUnlock(dur + 450L)
                setOnCompletionListener {
                    isCurrentlySpeaking = false
                }
                start()
            }
            true
        } catch (e: Exception) {
            isCurrentlySpeaking = false
            false
        }
    }

    private fun amplifyPcmSamples(rawPcm: ByteArray, gain: Float): ByteArray {
        if (gain <= 1.0f || rawPcm.size < 2) return rawPcm
        val out = ByteArray(rawPcm.size)
        val inBuf = ByteBuffer.wrap(rawPcm).order(ByteOrder.LITTLE_ENDIAN)
        val outBuf = ByteBuffer.wrap(out).order(ByteOrder.LITTLE_ENDIAN)

        while (inBuf.remaining() >= 2) {
            val sample = inBuf.short.toInt()
            val boosted = (sample * gain).toInt().coerceIn(Short.MIN_VALUE.toInt(), Short.MAX_VALUE.toInt())
            outBuf.putShort(boosted.toShort())
        }
        return out
    }

    private fun playRawPcm24kHz(pcmData: ByteArray, boostFactor: Float): Boolean {
        return try {
            stopCurrentAudio()
            val boostedData = amplifyPcmSamples(pcmData, boostFactor)
            val sampleRate = 24000
            val minBufSize = AudioTrack.getMinBufferSize(
                sampleRate,
                AudioFormat.CHANNEL_OUT_MONO,
                AudioFormat.ENCODING_PCM_16BIT
            )
            val bufferSize = kotlin.math.max(minBufSize, boostedData.size)

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
                .setBufferSizeInBytes(bufferSize)
                .setTransferMode(AudioTrack.MODE_STATIC)
                .build()

            track.setVolume(AudioTrack.getMaxVolume())

            try {
                loudnessEnhancer = LoudnessEnhancer(track.audioSessionId).apply {
                    setTargetGain(600)
                    enabled = true
                }
            } catch (e: Exception) {}

            val exactDurationMs = (boostedData.size * 1000L) / (sampleRate * 2)
            scheduleSpeakingUnlock(exactDurationMs + 450L)

            track.write(boostedData, 0, boostedData.size)
            track.play()
            currentAudioTrack = track
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
