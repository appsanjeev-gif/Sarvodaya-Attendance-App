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
import java.net.URL
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.security.MessageDigest
import java.util.Calendar
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicInteger
import kotlin.math.PI
import kotlin.math.exp
import kotlin.math.sin
import kotlin.random.Random

class RealisticVoiceManager(private val context: Context) {

    private val prefs = context.getSharedPreferences("SarvodayaVoicePrefs", Context.MODE_PRIVATE)
    private val mainHandler = Handler(Looper.getMainLooper())

    private val defPart1 = "AQ.Ab8RN6JboShANCf"
    private val defPart2 = "JrUUGoKliLkRUeWbYZ"
    private val defPart3 = "38ZEo9InfG-hroP9Q"

    private val builtInGeminiKeys = listOf(
        defPart1 + defPart2 + defPart3
    )

    private val geminiKeyIndex = AtomicInteger(0)
    private val elevenKeyIndex = AtomicInteger(0)

    fun getGeminiKeysString(): String {
        val dedicated = prefs.getString("custom_gemini_only_keys", null)
        if (dedicated != null) return dedicated
        val legacy = prefs.getString("custom_gemini_api_key", "") ?: ""
        return legacy.split(",", "\n", ";", " ")
            .map { it.trim() }
            .filter { it.isNotEmpty() && !it.startsWith("sk_", ignoreCase = true) }
            .joinToString("\n")
    }

    fun getElevenLabsKeysString(): String {
        val dedicated = prefs.getString("custom_eleven_only_keys", null)
        if (dedicated != null) return dedicated
        val legacy = prefs.getString("custom_gemini_api_key", "") ?: ""
        return legacy.split(",", "\n", ";", " ")
            .map { it.trim() }
            .filter { it.startsWith("sk_", ignoreCase = true) || (it.length >= 28 && !it.startsWith("AIza") && !it.startsWith("AQ.")) }
            .joinToString("\n")
    }

    fun saveSeparateApiKeys(geminiRaw: String, elevenRaw: String) {
        val gClean = geminiRaw.trim()
        val eClean = elevenRaw.trim()
        val combined = listOf(gClean, eClean).filter { it.isNotEmpty() }.joinToString("\n")

        prefs.edit()
            .putString("custom_gemini_only_keys", gClean)
            .putString("custom_eleven_only_keys", eClean)
            .putString("custom_gemini_api_key", combined)
            .apply()

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
        return prefs.getString("custom_gemini_api_key", "") ?: ""
    }

    fun saveCustomApiKey(newKey: String) {
        val tokens = newKey.split(",", "\n", ";", " ").map { it.trim() }.filter { it.length > 10 }
        val gKeys = tokens.filter { it.startsWith("AIza") || it.startsWith("AQ.") }.joinToString("\n")
        val eKeys = tokens.filter { !it.startsWith("AIza") && !it.startsWith("AQ.") }.joinToString("\n")
        saveSeparateApiKeys(gKeys, eKeys)
    }

    private fun getGeminiKeyPool(): List<String> {
        val raw = getGeminiKeysString()
        val list = raw.split(",", "\n", ";", " ")
            .map { it.trim() }
            .filter { it.length > 15 && !it.startsWith("sk_", ignoreCase = true) }
        val hasCustomEleven = getElevenLabsKeyPool().isNotEmpty()
        return when {
            list.isNotEmpty() -> list
            hasCustomEleven -> emptyList()
            else -> builtInGeminiKeys
        }
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
                "JBFqnCBsd6RMkjVDRZzb",
                "pNInz6obpgDQGcFmaJgB",
                "onwK4e9ZLuTAKqWW03F9"
            ),
            stylePrompt = "Speak like an energetic, proud, warm Indian Hindi radio RJ host with a natural smile and lively pace:"
        ),
        AiVoiceProfile(
            name = "Autonoe",
            elevenVoiceIds = listOf(
                "EXAVITQu4vr4xnSDxMaL",
                "Xb7hH8MSUJpSbSDYk0k2",
                "21m00Tcm4TlvDq8ikWAM"
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
        File(context.getExternalFilesDir(null) ?: context.filesDir, "ai_studio_voice_bank").apply {
            if (!exists()) mkdirs()
        }
    }

    init {
        maintainTwoHourRollingWelcomeCache()
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
            val (pcmBytes, code) = fetchGoogleAiStudioTtsPcm(welcomeText, profile, geminiPool[idx])
            if (pcmBytes != null && pcmBytes.size > 2000) {
                writeFileAtomically(targetPcm, pcmBytes)
                return true
            }
            if (code != 429 && code != 400 && code != 403) break
        }

        val elevenPool = getElevenLabsKeyPool()
        for (attempt in elevenPool.indices) {
            val idx = (elevenKeyIndex.getAndIncrement() % elevenPool.size).let { if (it < 0) it + elevenPool.size else it }
            val (mp3Bytes, code) = fetchElevenLabsMp3WithFallback(welcomeText, profile.elevenVoiceIds, elevenPool[idx])
            if (mp3Bytes != null && mp3Bytes.size > 1000) {
                writeFileAtomically(targetMp3, mp3Bytes)
                return true
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
            "प्रेरणादायक और जोशपूर्ण (Motivational & Energetic)",
            "हाई-टेक डिजिटल स्मार्ट कैंपस (Modern Digital Campus)",
            "बच्चों और शिक्षकों से आत्मीय संवाद (Warm Student-Teacher Connect)",
            "अनुशासन, मुस्कान और गर्व से भरा (Proud & Disciplined)"
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
                    नियम: केवल 18 से 24 शब्दों में एक नया, आकर्षक रेडियो एंकर जैसा हिंदी स्वागत वाक्य लिखें जो छात्रों को कैमरे के सामने मुस्कुराकर स्मार्ट उपस्थिति दर्ज कराने के लिए प्रेरित करे। केवल शुद्ध वाक्य लौटाएं, कोई कोट्स या अतिरिक्त शब्द न लिखें। (Slot: $slotId)
                """.trimIndent()

                val payload = JSONObject().apply {
                    put("contents", JSONArray().put(
                        JSONObject().put("parts", JSONArray().put(JSONObject().put("text", prompt)))
                    ))
                }

                conn.outputStream.use { it.write(payload.toString().toByteArray(Charsets.UTF_8)) }
                if (conn.responseCode == 200) {
                    val resp = conn.inputStream.bufferedReader(Charsets.UTF_8).use { it.readText() }
                    val text = JSONObject(resp)
                        .optJSONArray("candidates")?.optJSONObject(0)
                        ?.optJSONObject("content")?.optJSONArray("parts")?.optJSONObject(0)
                        ?.optString("text", "")?.trim()
                        ?.replace("\"", "")?.replace("*", "")?.replace("\n", " ") ?: ""
                    if (text.length in 25..220 && text.contains("सर्वोदय")) {
                        return text
                    }
                }
            } catch (e: Exception) {}
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
            if (playRawPcm24kHz(file1.readBytes(), boostFactor)) return
        }
        if (file2.exists() && file2.length() > 2000) {
            if (playRawPcm24kHz(file2.readBytes(), boostFactor)) return
        }
        if (mp3CacheFile.exists() && mp3CacheFile.length() > 1000) {
            if (playMp3File(mp3CacheFile)) return
        }

        val geminiPool = getGeminiKeyPool()
        if (geminiPool.isNotEmpty()) {
            val poolSize = geminiPool.size
            for (attempt in 0 until poolSize) {
                val idx = (geminiKeyIndex.getAndIncrement() % poolSize).let { if (it < 0) it + poolSize else it }
                val candidateKey = geminiPool[idx]
                val (pcmBytes, errCode) = fetchGoogleAiStudioTtsPcm(text, chosenVoice, candidateKey)
                if (pcmBytes != null && pcmBytes.size > 2000) {
                    writeFileAtomically(file1, pcmBytes)
                    if (playRawPcm24kHz(pcmBytes, boostFactor)) return
                }
                if (errCode != 429 && errCode != 400 && errCode != 403) break
            }
        }

        val elevenPool = getElevenLabsKeyPool()
        if (elevenPool.isNotEmpty()) {
            val poolSize = elevenPool.size
            for (attempt in 0 until poolSize) {
                val idx = (elevenKeyIndex.getAndIncrement() % poolSize).let { if (it < 0) it + poolSize else it }
                val candidateKey = elevenPool[idx]
                val (mp3Bytes, errCode) = fetchElevenLabsMp3WithFallback(text, chosenVoice.elevenVoiceIds, candidateKey)
                if (mp3Bytes != null && mp3Bytes.size > 1000) {
                    writeFileAtomically(mp3CacheFile, mp3Bytes)
                    if (playMp3File(mp3CacheFile)) return
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
    ): Pair<ByteArray?, Int> {
        val models = listOf("gemini-2.5-flash-preview-tts", "gemini-2.5-pro-preview-tts")
        var lastHttpCode = 0

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
                        return Pair(Base64.decode(b64Audio, Base64.DEFAULT), 200)
                    }
                } else if (lastHttpCode == 400 || lastHttpCode == 403 || lastHttpCode == 429) {
                    break
                }
            } catch (e: Exception) {}
        }
        return Pair(null, lastHttpCode)
    }

    private fun fetchElevenLabsMp3WithFallback(
        hindiText: String,
        voiceIds: List<String>,
        elevenApiKey: String
    ): Pair<ByteArray?, Int> {
        val cleanKey = elevenApiKey.trim()
        val models = listOf("eleven_multilingual_v2", "eleven_turbo_v2_5")
        var lastCode = 0

        for (vId in voiceIds) {
            for (modelId in models) {
                try {
                    val endpoint = "https://api.elevenlabs.io/v1/text-to-speech/$vId?output_format=mp3_44100_128"
                    val conn = (URL(endpoint).openConnection() as HttpURLConnection).apply {
                        requestMethod = "POST"
                        connectTimeout = 10000
                        readTimeout = 15000
                        doOutput = true
                        setRequestProperty("xi-api-key", cleanKey)
                        setRequestProperty("Content-Type", "application/json; charset=UTF-8")
                        setRequestProperty("Accept", "audio/mpeg")
                    }

                    val payload = JSONObject().apply {
                        put("text", hindiText)
                        put("model_id", modelId)
                        put("voice_settings", JSONObject().apply {
                            put("stability", 0.50)
                            put("similarity_boost", 0.80)
                        })
                    }

                    conn.outputStream.use { os ->
                        os.write(payload.toString().toByteArray(Charsets.UTF_8))
                    }

                    lastCode = conn.responseCode
                    if (lastCode == 200) {
                        val bytes = conn.inputStream.readBytes()
                        if (bytes.size > 1000) {
                            return Pair(bytes, 200)
                        }
                    } else if (lastCode == 401 || lastCode == 429) {
                        return Pair(null, lastCode)
                    }
                } catch (e: Exception) {}
            }
        }
        return Pair(null, lastCode)
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
