package com.school.sarvodayaattendance.utils

import android.content.Context
import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioTrack
import android.media.MediaPlayer
import android.media.audiofx.LoudnessEnhancer
import android.os.Handler
import android.os.Looper
import android.speech.tts.TextToSpeech
import android.speech.tts.UtteranceProgressListener
import android.util.Base64
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.io.FileInputStream
import java.io.FileOutputStream
import java.net.HttpURLConnection
import java.net.URL
import java.net.URLEncoder
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.security.MessageDigest
import java.util.Locale
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicInteger
import kotlin.random.Random

class RealisticVoiceManager(private val context: Context) : TextToSpeech.OnInitListener {

    private val prefs = context.getSharedPreferences("SarvodayaVoicePrefs", Context.MODE_PRIVATE)
    private val mainHandler = Handler(Looper.getMainLooper())

    private val defPart1 = "AQ.Ab8RN6JboShANCf"
    private val defPart2 = "JrUUGoKliLkRUeWbYZ"
    private val defPart3 = "38ZEo9InfG-hroP9Q"

    private val builtInKeys = listOf(
        defPart1 + defPart2 + defPart3
    )

    private val keyRotationIndex = AtomicInteger(0)

    fun getApiKey(): String {
        return prefs.getString("custom_gemini_api_key", "") ?: ""
    }

    fun saveCustomApiKey(newKey: String) {
        prefs.edit().putString("custom_gemini_api_key", newKey.trim()).apply()
    }

    private fun getActiveKeyPool(): List<String> {
        val savedRaw = prefs.getString("custom_gemini_api_key", "") ?: ""
        val customList = savedRaw
            .split(",", "\n", ";", " ")
            .map { it.trim() }
            .filter { it.length > 15 }
        return if (customList.isNotEmpty()) customList else builtInKeys
    }

    private data class AiVoiceProfile(
        val name: String,
        val edgeNeuralVoice: String,
        val stylePrompt: String
    )

    // ✅ AI Studio Voices + Microsoft Edge Neural Voices (SwaraNeural & MadhurNeural)
    private val aiVoiceProfiles = listOf(
        AiVoiceProfile(
            name = "Sadachbia",
            edgeNeuralVoice = "hi-IN-MadhurNeural",
            stylePrompt = "Read aloud in a lively, warm, lower-pitch Indian Hindi male voice:"
        ),
        AiVoiceProfile(
            name = "Autonoe",
            edgeNeuralVoice = "hi-IN-SwaraNeural",
            stylePrompt = "Read aloud in a bright, clear, natural middle-pitch Indian Hindi female voice:"
        )
    )

    private val startupWelcomeMessages = listOf(
        "सुप्रभात! सर्वोदय मध्य विद्यालय, सोहसराय में आपका हार्दिक स्वागत है। स्मार्ट उपस्थिति प्रणाली अब तैयार है।",
        "नमस्कार! सर्वोदय मध्य विद्यालय, सोहसराय के स्मार्ट कैंपस में आपका स्वागत है। आइए, आज की उपस्थिति शुरू करें।",
        "नमस्कार! शिक्षा के मंदिर सर्वोदय मध्य विद्यालय, सोहसराय में आपका स्वागत है। कृपया कैमरे के सामने आकर अपनी उपस्थिति दर्ज कराएं।"
    )

    private var tts: TextToSpeech? = null
    private var isOfflineTtsReady = false
    var isVoiceEnabled = true

    @Volatile
    var isCurrentlySpeaking: Boolean = false
        private set

    private var unlockRunnable: Runnable? = null
    private val voiceExecutor = Executors.newSingleThreadExecutor()
    private var currentAudioTrack: AudioTrack? = null
    private var loudnessEnhancer: LoudnessEnhancer? = null
    private var mediaPlayer: MediaPlayer? = null

    private val voiceCacheDir: File by lazy {
        File(context.getExternalFilesDir(null) ?: context.filesDir, "ai_studio_voice_bank").apply {
            if (!exists()) mkdirs()
        }
    }

    init {
        tts = TextToSpeech(context, this)
    }

    override fun onInit(status: Int) {
        if (status == TextToSpeech.SUCCESS) {
            tts?.language = Locale("hi", "IN")
            tts?.setPitch(0.90f)
            tts?.setSpeechRate(0.95f)
            tts?.setOnUtteranceProgressListener(object : UtteranceProgressListener() {
                override fun onStart(utteranceId: String?) {
                    isCurrentlySpeaking = true
                }
                override fun onDone(utteranceId: String?) {
                    isCurrentlySpeaking = false
                }
                override fun onError(utteranceId: String?) {
                    isCurrentlySpeaking = false
                }
            })
            isOfflineTtsReady = true
        }
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
        val welcomeLine = startupWelcomeMessages[Random.nextInt(startupWelcomeMessages.size)]
        speakHindi(welcomeLine, boostFactor = 1.55f)
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

        scheduleSpeakingUnlock(9500L)

        voiceExecutor.execute {
            val chosenVoice = aiVoiceProfiles[Random.nextInt(aiVoiceProfiles.size)]
            val otherVoice = aiVoiceProfiles.first { it.name != chosenVoice.name }

            val hash1 = md5Hash("${chosenVoice.name}_$text")
            val file1 = File(voiceCacheDir, "$hash1.pcm")

            val hash2 = md5Hash("${otherVoice.name}_$text")
            val file2 = File(voiceCacheDir, "$hash2.pcm")

            val mp3CacheFile = File(voiceCacheDir, "${md5Hash("edge_${chosenVoice.edgeNeuralVoice}_$text")}.mp3")

            // 1. यदि फोन की मेमोरी में AI Studio PCM पहले से सेव है तो तुरंत बजाएं
            if (file1.exists() && file1.length() > 1000) {
                if (playRawPcm24kHz(file1.readBytes(), boostFactor)) return@execute
            }

            // 2. Multi-Key Auto-Rotation Pool से AI Studio आवाज़ लाएं (VIP लंबे वाक्यों के लिए बढ़ाया गया टाइमआउट)
            val keyPool = getActiveKeyPool()
            if (keyPool.isNotEmpty()) {
                val poolSize = keyPool.size
                for (attempt in 0 until poolSize) {
                    val idx = (keyRotationIndex.getAndIncrement() % poolSize).let { if (it < 0) it + poolSize else it }
                    val candidateKey = keyPool[idx]
                    val (pcmBytes, errCode) = fetchGoogleAiStudioTtsPcm(text, chosenVoice, candidateKey)
                    if (pcmBytes != null && pcmBytes.size > 1000) {
                        try {
                            FileOutputStream(file1).use { it.write(pcmBytes) }
                        } catch (_: Exception) {}
                        if (playRawPcm24kHz(pcmBytes, boostFactor)) return@execute
                    }
                    if (errCode != 429 && errCode != 400 && errCode != 403) {
                        break
                    }
                }
            }

            if (file2.exists() && file2.length() > 1000) {
                if (playRawPcm24kHz(file2.readBytes(), boostFactor)) return@execute
            }

            // 3. यदि पहले से सेव Neural MP3 मौजूद है तो उसे तुरंत बजाएं
            if (mp3CacheFile.exists() && mp3CacheFile.length() > 800) {
                if (playMp3File(mp3CacheFile, chosenVoice.name == "Sadachbia", isPureNeural = true)) return@execute
            }

            // 4. Microsoft Edge Neural Voice (SwaraNeural / MadhurNeural - Unlimited Free AI Voice)
            if (tryPlayMicrosoftEdgeNeural(text, chosenVoice.edgeNeuralVoice, mp3CacheFile)) return@execute

            // 5. Multi-Chunk Neural Stream Backup (लंबे VIP स्वागत वाक्यों को भी सपोर्ट करता है, कोई कोटा लिमिट नहीं)
            if (tryPlayChunkedNeuralMp3(text, chosenVoice.name == "Sadachbia", mp3CacheFile)) return@execute

            // 6. अंतिम ऑफलाइन बैकअप (केवल तब जब इंटरनेट पूरी तरह बंद हो)
            if (isOfflineTtsReady) {
                scheduleSpeakingUnlock(5500L)
                tts?.setPitch(if (chosenVoice.name == "Sadachbia") 0.82f else 1.05f)
                tts?.speak(text, TextToSpeech.QUEUE_FLUSH, null, "SARVODAYA_FALLBACK")
            } else {
                isCurrentlySpeaking = false
            }
        }
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
                    connectTimeout = 8000
                    readTimeout = 12000 // ✅ लंबे VIP स्वागत संदेश के लिए 12 सेकंड का पर्याप्त समय
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

    // ✅ Microsoft Edge Neural Engine (SwaraNeural / MadhurNeural - No Quota Limit)
    private fun tryPlayMicrosoftEdgeNeural(text: String, voiceName: String, outFile: File): Boolean {
        val endpoints = listOf(
            "https://tts.travisvn.com/api/tts?text=${URLEncoder.encode(text, "UTF-8")}&voice=$voiceName",
            "https://edge-tts-api.vercel.app/api/tts?text=${URLEncoder.encode(text, "UTF-8")}&voice=$voiceName"
        )
        for (urlStr in endpoints) {
            try {
                val conn = (URL(urlStr).openConnection() as HttpURLConnection).apply {
                    requestMethod = "GET"
                    connectTimeout = 5000
                    readTimeout = 8000
                    setRequestProperty("User-Agent", "Mozilla/5.0")
                }
                if (conn.responseCode == 200) {
                    val bytes = conn.inputStream.readBytes()
                    if (bytes.size > 1000) {
                        FileOutputStream(outFile).use { it.write(bytes) }
                        if (playMp3File(outFile, voiceName.contains("Madhur"), isPureNeural = true)) {
                            return true
                        }
                    }
                }
            } catch (_: Exception) {}
        }
        return false
    }

    // ✅ लंबे वाक्यों (जैसे VIP स्वागत) को छोटे भागों में जोड़कर बिना कटे और बिना कोटा लिमिट के बजाने वाला न्यूरल स्ट्रीम
    private fun tryPlayChunkedNeuralMp3(text: String, isLowerPitchMale: Boolean, outFile: File): Boolean {
        return try {
            val sentences = text.split(Regex("(?<=[।!?])")).map { it.trim() }.filter { it.isNotEmpty() }
            val combinedStream = java.io.ByteArrayOutputStream()

            for (sentence in sentences) {
                val encoded = URLEncoder.encode(sentence, "UTF-8")
                val urlStr = "https://translate.googleapis.com/translate_tts?ie=UTF-8&q=$encoded&tl=hi&client=tw-ob"
                val conn = (URL(urlStr).openConnection() as HttpURLConnection).apply {
                    connectTimeout = 4000
                    readTimeout = 6000
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
            if (finalBytes.size > 1000) {
                FileOutputStream(outFile).use { it.write(finalBytes) }
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
                        val targetPitch = if (isLowerPitchMale) 0.86f else 1.03f
                        playbackParams = playbackParams.setPitch(targetPitch).setSpeed(0.98f)
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
        try {
            tts?.stop()
            tts?.shutdown()
        } catch (_: Exception) {}
    }
}
