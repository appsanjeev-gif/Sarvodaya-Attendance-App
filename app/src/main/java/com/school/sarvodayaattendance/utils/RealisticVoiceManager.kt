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
import java.io.ByteArrayOutputStream
import java.io.File
import java.io.FileInputStream
import java.io.FileOutputStream
import java.net.HttpURLConnection
import java.net.URL
import java.net.URLEncoder
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.security.MessageDigest
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicInteger
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

    fun getApiKey(): String {
        return prefs.getString("custom_gemini_api_key", "") ?: ""
    }

    fun saveCustomApiKey(newKey: String) {
        prefs.edit().putString("custom_gemini_api_key", newKey.trim()).apply()
        preWarmStartupWelcomesInBackground()
    }

    private fun parseAllSavedTokens(): List<String> {
        val savedRaw = prefs.getString("custom_gemini_api_key", "") ?: ""
        return savedRaw
            .split(",", "\n", ";", " ")
            .map { it.trim() }
            .filter { it.length > 15 }
    }

    private fun getGeminiKeyPool(): List<String> {
        val all = parseAllSavedTokens()
        val geminiList = all.filter {
            it.startsWith("AQ.", ignoreCase = true) ||
            it.startsWith("AIza", ignoreCase = true) ||
            (!it.startsWith("sk_", ignoreCase = true) && it.length >= 35)
        }
        return if (geminiList.isNotEmpty()) geminiList else builtInGeminiKeys
    }

    private fun getElevenLabsKeyPool(): List<String> {
        val all = parseAllSavedTokens()
        return all.filter {
            it.startsWith("sk_", ignoreCase = true) ||
            (it.length == 32 && it.all { ch -> ch.isLetterOrDigit() })
        }
    }

    private data class AiVoiceProfile(
        val name: String,
        val elevenVoiceId: String,
        val stylePrompt: String
    )

    private val aiVoiceProfiles = listOf(
        AiVoiceProfile(
            name = "Sadachbia",
            elevenVoiceId = "pNInz6obpgDQGcFmaJgB",
            stylePrompt = "Read aloud in a lively, warm, lower-pitch Indian Hindi male voice:"
        ),
        AiVoiceProfile(
            name = "Autonoe",
            elevenVoiceId = "EXAVITQu4vr4xnSDxMaL",
            stylePrompt = "Read aloud in a bright, clear, natural middle-pitch Indian Hindi female voice:"
        )
    )

    private val startupWelcomeMessages = listOf(
        "सुप्रभात! सर्वोदय मध्य विद्यालय, सोहसराय में आपका हार्दिक स्वागत है। स्मार्ट उपस्थिति प्रणाली अब तैयार है।",
        "नमस्कार! सर्वोदय मध्य विद्यालय, सोहसराय के स्मार्ट कैंपस में आपका स्वागत है। आइए, आज की उपस्थिति शुरू करें।",
        "नमस्कार! शिक्षा के मंदिर सर्वोदय मध्य विद्यालय, सोहसराय में आपका स्वागत है। कृपया कैमरे के सामने आकर अपनी उपस्थिति दर्ज कराएं।"
    )

    var isVoiceEnabled = true

    @Volatile
    var isCurrentlySpeaking: Boolean = false
        private set

    private var unlockRunnable: Runnable? = null
    private val voiceExecutor = Executors.newSingleThreadExecutor()
    private val preWarmExecutor = Executors.newSingleThreadExecutor()
    private var currentAudioTrack: AudioTrack? = null
    private var loudnessEnhancer: LoudnessEnhancer? = null
    private var mediaPlayer: MediaPlayer? = null

    private val voiceCacheDir: File by lazy {
        File(context.getExternalFilesDir(null) ?: context.filesDir, "ai_studio_voice_bank").apply {
            if (!exists()) mkdirs()
        }
    }

    init {
        preWarmStartupWelcomesInBackground()
    }

    // ✅ ऐप खुलते ही बैकग्राउंड में तीनों स्वागत संदेशों को स्थायी रूप से फोन में सेव कर लेता है
    private fun preWarmStartupWelcomesInBackground() {
        preWarmExecutor.execute {
            try {
                for (i in startupWelcomeMessages.indices) {
                    val msg = startupWelcomeMessages[i]
                    val fixedPcm = File(voiceCacheDir, "startup_welcome_fixed_$i.pcm")
                    val fixedMp3 = File(voiceCacheDir, "startup_welcome_fixed_$i.mp3")
                    if ((fixedPcm.exists() && fixedPcm.length() > 2000) ||
                        (fixedMp3.exists() && fixedMp3.length() > 1500)) {
                        continue
                    }
                    val profile = aiVoiceProfiles[i % aiVoiceProfiles.size]
                    val geminiPool = getGeminiKeyPool()
                    for (attempt in geminiPool.indices) {
                        val idx = (geminiKeyIndex.getAndIncrement() % geminiPool.size).let { if (it < 0) it + geminiPool.size else it }
                        val (pcm, code) = fetchGoogleAiStudioTtsPcm(msg, profile, geminiPool[idx])
                        if (pcm != null && pcm.size > 2000) {
                            writeFileAtomically(fixedPcm, pcm)
                            break
                        }
                        if (code != 429 && code != 400 && code != 403) break
                    }
                }
            } catch (_: Exception) {}
        }
    }

    private fun writeFileAtomically(targetFile: File, bytes: ByteArray) {
        try {
            val tempFile = File(targetFile.parentFile, "${targetFile.name}.tmp")
            FileOutputStream(tempFile).use { fos ->
                fos.write(bytes)
                fos.fd.sync()
            }
            if (tempFile.exists() && tempFile.length() > 1000) {
                tempFile.renameTo(targetFile)
            }
        } catch (_: Exception) {}
    }

    private fun scheduleSpeakingUnlock(durationMs: Long) {
        isCurrentlySpeaking = true
        unlockRunnable?.let { mainHandler.removeCallbacks(it) }
        val r = Runnable { isCurrentlySpeaking = false }
        unlockRunnable = r
        mainHandler.postDelayed(r, durationMs)
    }

    fun speakStartupWelcome() {
        if (!isVoiceEnabled) return
        voiceExecutor.execute {
            val readyIndices = startupWelcomeMessages.indices.filter { idx ->
                File(voiceCacheDir, "startup_welcome_fixed_$idx.pcm").let { it.exists() && it.length() > 2000 } ||
                File(voiceCacheDir, "startup_welcome_fixed_$idx.mp3").let { it.exists() && it.length() > 1500 }
            }

            val chosenIdx = if (readyIndices.isNotEmpty()) {
                readyIndices[Random.nextInt(readyIndices.size)]
            } else {
                Random.nextInt(startupWelcomeMessages.size)
            }

            val fixedPcm = File(voiceCacheDir, "startup_welcome_fixed_$chosenIdx.pcm")
            if (fixedPcm.exists() && fixedPcm.length() > 2000) {
                if (playRawPcm24kHz(fixedPcm.readBytes(), 1.55f)) return@execute
            }

            val fixedMp3 = File(voiceCacheDir, "startup_welcome_fixed_$chosenIdx.mp3")
            if (fixedMp3.exists() && fixedMp3.length() > 1500) {
                if (playMp3File(fixedMp3, isLowerPitchMale = false, isPureNeural = true)) return@execute
            }

            val welcomeLine = startupWelcomeMessages[chosenIdx]
            speakHindiInternal(welcomeLine, boostFactor = 1.55f, fixedStartupIndex = chosenIdx)
        }
    }

    fun speakVipInspectionWelcome(presentCount: Int, totalCount: Int) {
        if (!isVoiceEnabled) return
        val msg = "नमस्कार! सर्वोदय मध्य विद्यालय, सोहसराय के स्मार्ट एआई परिसर में माननीय अधिकारी एवं अतिथि महोदय का हार्दिक अभिनंदन है। आज विद्यालय में कुल $totalCount में से $presentCount छात्र उपस्थित हैं।"
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
            speakHindiInternal(text, boostFactor, fixedStartupIndex = -1)
        }
    }

    private fun speakHindiInternal(text: String, boostFactor: Float, fixedStartupIndex: Int) {
        scheduleSpeakingUnlock(10000L)

        val chosenVoice = aiVoiceProfiles[Random.nextInt(aiVoiceProfiles.size)]
        val otherVoice = aiVoiceProfiles.first { it.name != chosenVoice.name }

        val hash1 = md5Hash("${chosenVoice.name}_$text")
        val file1 = if (fixedStartupIndex >= 0) {
            File(voiceCacheDir, "startup_welcome_fixed_$fixedStartupIndex.pcm")
        } else {
            File(voiceCacheDir, "$hash1.pcm")
        }

        val hash2 = md5Hash("${otherVoice.name}_$text")
        val file2 = File(voiceCacheDir, "$hash2.pcm")

        val mp3CacheFile = if (fixedStartupIndex >= 0) {
            File(voiceCacheDir, "startup_welcome_fixed_$fixedStartupIndex.mp3")
        } else {
            File(voiceCacheDir, "${md5Hash("ai_mp3_$text")}.mp3")
        }

        // 1. पहले से सेव AI Studio PCM चेक करें
        if (file1.exists() && file1.length() > 2000) {
            if (playRawPcm24kHz(file1.readBytes(), boostFactor)) return
        }
        if (file2.exists() && file2.length() > 2000) {
            if (playRawPcm24kHz(file2.readBytes(), boostFactor)) return
        }
        if (mp3CacheFile.exists() && mp3CacheFile.length() > 1500) {
            if (playMp3File(mp3CacheFile, chosenVoice.name == "Sadachbia", isPureNeural = true)) return
        }

        // 2. Multi-Gemini Key Pool (AQ.Ab8... / AIza...)
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
                if (errCode != 429 && errCode != 400 && errCode != 403) {
                    break
                }
            }
        }

        // 3. Multi-ElevenLabs Key Pool (sk_...)
        val elevenPool = getElevenLabsKeyPool()
        if (elevenPool.isNotEmpty()) {
            val poolSize = elevenPool.size
            for (attempt in 0 until poolSize) {
                val idx = (elevenKeyIndex.getAndIncrement() % poolSize).let { if (it < 0) it + poolSize else it }
                val candidateKey = elevenPool[idx]
                val (mp3Bytes, errCode) = fetchElevenLabsMp3(text, chosenVoice.elevenVoiceId, candidateKey)
                if (mp3Bytes != null && mp3Bytes.size > 1500) {
                    writeFileAtomically(mp3CacheFile, mp3Bytes)
                    if (playMp3File(mp3CacheFile, chosenVoice.name == "Sadachbia", isPureNeural = true)) return
                }
                if (errCode != 429 && errCode != 401 && errCode != 403) {
                    break
                }
            }
        }

        // 4. Online Natural Neural Stream (No Quota Limit, Never Boring System TTS)
        if (tryPlayChunkedNeuralMp3(text, chosenVoice.name == "Sadachbia", mp3CacheFile)) return

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
            } catch (_: Exception) {}
        }
        return Pair(null, lastHttpCode)
    }

    private fun fetchElevenLabsMp3(
        hindiText: String,
        voiceId: String,
        elevenApiKey: String
    ): Pair<ByteArray?, Int> {
        var httpCode = 0
        try {
            val endpoint = "https://api.elevenlabs.io/v1/text-to-speech/$voiceId"
            val conn = (URL(endpoint).openConnection() as HttpURLConnection).apply {
                requestMethod = "POST"
                connectTimeout = 8000
                readTimeout = 12000
                doOutput = true
                setRequestProperty("xi-api-key", elevenApiKey)
                setRequestProperty("Content-Type", "application/json; charset=UTF-8")
                setRequestProperty("Accept", "audio/mpeg")
            }

            val payload = JSONObject().apply {
                put("text", hindiText)
                put("model_id", "eleven_multilingual_v2")
                put("voice_settings", JSONObject().apply {
                    put("stability", 0.55)
                    put("similarity_boost", 0.80)
                })
            }

            conn.outputStream.use { os ->
                os.write(payload.toString().toByteArray(Charsets.UTF_8))
            }

            httpCode = conn.responseCode
            if (httpCode == 200) {
                val bytes = conn.inputStream.readBytes()
                if (bytes.size > 1500) {
                    return Pair(bytes, 200)
                }
            }
        } catch (_: Exception) {}
        return Pair(null, httpCode)
    }

    private fun tryPlayChunkedNeuralMp3(text: String, isLowerPitchMale: Boolean, outFile: File): Boolean {
        return try {
            val sentences = text.split(Regex("(?<=[।!?])")).map { it.trim() }.filter { it.isNotEmpty() }
            val combinedStream = ByteArrayOutputStream()

            for (sentence in sentences) {
                val encoded = URLEncoder.encode(sentence, "UTF-8")
                val urlStr = "https://translate.googleapis.com/translate_tts?ie=UTF-8&q=$encoded&tl=hi&client=tw-ob"
                val conn = (URL(urlStr).openConnection() as HttpURLConnection).apply {
                    connectTimeout = 5000
                    readTimeout = 7000
                    setRequestProperty("User-Agent", "Mozilla/5.0")
                }
                if (conn.responseCode == 200) {
                    val chunkBytes = conn.inputStream.readBytes()
                    if (chunkBytes.isNotEmpty()) {
                        combinedStream.write(chunkBytes)
                    }
                }
            }

            val finalBytes = combinedStream.toByteArray()
            if (finalBytes.size > 1500) {
                writeFileAtomically(outFile, finalBytes)
                playMp3File(outFile, isLowerPitchMale, isPureNeural = false)
            } else {
                false
            }
        } catch (e: Exception) {
            false
        }
    }

    private fun playMp3File(mp3File: File, isLowerPitchMale: Boolean, isPureNeural: Boolean): Boolean {
        return try {
            stopCurrentAudio()
            FileInputStream(mp3File).use { fis ->
                mediaPlayer = MediaPlayer().apply {
                    setAudioAttributes(
                        AudioAttributes.Builder()
                            .setUsage(AudioAttributes.USAGE_MEDIA)
                            .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH)
                            .build()
                    )
                    setDataSource(fis.fd)
                    prepare()
                    setVolume(1.0f, 1.0f)
                    try {
                        loudnessEnhancer = LoudnessEnhancer(audioSessionId).apply {
                            setTargetGain(650)
                            enabled = true
                        }
                    } catch (_: Exception) {}
                    if (!isPureNeural && android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.M) {
                        val targetPitch = if (isLowerPitchMale) 0.90f else 1.02f
                        playbackParams = playbackParams.setPitch(targetPitch).setSpeed(1.0f)
                    }
                    val dur = duration.toLong().coerceAtLeast(3500L)
                    scheduleSpeakingUnlock(dur + 450L)
                    setOnCompletionListener {
                        isCurrentlySpeaking = false
                    }
                    start()
                }
            }
            true
        } catch (e: Exception) {
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
            } catch (_: Exception) {}

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
        } catch (_: Exception) {}

        try {
            currentAudioTrack?.let {
                if (it.playState == AudioTrack.PLAYSTATE_PLAYING) it.stop()
                it.release()
            }
            currentAudioTrack = null
        } catch (_: Exception) {}

        try {
            mediaPlayer?.let {
                if (it.isPlaying) it.stop()
                it.release()
            }
            mediaPlayer = null
        } catch (_: Exception) {}
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
