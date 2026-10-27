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
import android.widget.Toast
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
import kotlin.random.Random

class RealisticVoiceManager(private val context: Context) : TextToSpeech.OnInitListener {

    private val prefs = context.getSharedPreferences("SarvodayaVoicePrefs", Context.MODE_PRIVATE)
    private val mainHandler = Handler(Looper.getMainLooper())

    private val defPart1 = "AQ.Ab8RN6JboShANCf"
    private val defPart2 = "JrUUGoKliLkRUeWbYZ"
    private val defPart3 = "38ZEo9InfG-hroP9Q"

    fun getApiKey(): String {
        val saved = prefs.getString("custom_gemini_api_key", "") ?: ""
        return if (saved.isNotBlank()) saved.trim() else (defPart1 + defPart2 + defPart3)
    }

    fun saveCustomApiKey(newKey: String) {
        prefs.edit().putString("custom_gemini_api_key", newKey.trim()).apply()
    }

    private data class AiVoiceProfile(
        val name: String,
        val stylePrompt: String
    )

    private val aiVoiceProfiles = listOf(
        AiVoiceProfile(
            name = "Sadachbia", // Male: Lively, Lower pitch
            stylePrompt = "Read aloud in a lively, warm, lower-pitch Indian Hindi male voice:"
        ),
        AiVoiceProfile(
            name = "Autonoe",   // Female: Bright, Middle pitch
            stylePrompt = "Read aloud in a bright, clear, natural middle-pitch Indian Hindi female voice:"
        )
    )

    // ✅ आपके द्वारा चुने गए स्वागत वाक्य (विकल्प 2, 3 और 5 - रैंडम मोड)
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

    // ✅ ऐप स्टार्ट होने पर हर बार तेज़ आवाज़ में रैंडम स्वागत वाक्य बोलने का फंक्शन
    fun speakStartupWelcome() {
        if (!isVoiceEnabled) return
        val welcomeLine = startupWelcomeMessages[Random.nextInt(startupWelcomeMessages.size)]
        speakHindi(welcomeLine, boostFactor = 1.55f)
    }

    fun speakGreeting(type: String, studentName: String) {
        if (!isVoiceEnabled) return
        val cleanName = studentName.trim()
        // ✅ हास्यास्पद वाक्य के लिए केवल फर्स्ट नेम (First Name) निकालें
        val firstName = cleanName.substringBefore(" ").ifBlank { cleanName }

        val message = when (type) {
            "success" -> "स्वागत है $cleanName! आपकी आज की उपस्थिति दर्ज हो चुकी है।"
            "warning" -> "ध्यान दें $cleanName, आपकी आज की हाज़िरी पहले ही लग चुकी है।"
            "move_away" -> "चल हट भी जा अब $firstName, दूसरे को भी अटेंडेंस बनाने का मौका दो।"
            else -> "क्षमा करें, चेहरा पहचाना नहीं जा सका, कृपया कैमरे के सामने सीधे देखें।"
        }
        speakHindi(message, boostFactor = 1.35f)
    }

    fun speakHindi(text: String, boostFactor: Float = 1.35f) {
        if (!isVoiceEnabled || text.isBlank()) return

        scheduleSpeakingUnlock(6000L)

        voiceExecutor.execute {
            val chosenVoice = aiVoiceProfiles[Random.nextInt(aiVoiceProfiles.size)]
            val otherVoice = aiVoiceProfiles.first { it.name != chosenVoice.name }

            val hash1 = md5Hash("${chosenVoice.name}_$text")
            val file1 = File(voiceCacheDir, "$hash1.pcm")

            val hash2 = md5Hash("${otherVoice.name}_$text")
            val file2 = File(voiceCacheDir, "$hash2.pcm")

            // 1. यदि चुनी गई आवाज़ पहले से फोन में सेव है, तो तुरंत तेज़ आवाज़ में बजाएं
            if (file1.exists() && file1.length() > 1000) {
                if (playRawPcm24kHz(file1.readBytes(), boostFactor)) return@execute
            }

            // 2. Google AI Studio से नई आवाज़ मंगाएं
            val apiKey = getApiKey()
            if (apiKey.isNotBlank()) {
                val (pcmBytes, errCode) = fetchGoogleAiStudioTtsPcm(text, chosenVoice, apiKey)
                if (pcmBytes != null && pcmBytes.size > 1000) {
                    try {
                        FileOutputStream(file1).use { it.write(pcmBytes) }
                    } catch (_: Exception) {}
                    if (playRawPcm24kHz(pcmBytes, boostFactor)) return@execute
                } else {
                    if (file2.exists() && file2.length() > 1000) {
                        if (playRawPcm24kHz(file2.readBytes(), boostFactor)) return@execute
                    }
                    if (errCode == 429) {
                        showDebugToast("⚠️ AI Studio का प्रति-मिनट कोटा पूर्ण (429)! 1 मिनट बाद प्रयास करें।")
                    } else if (errCode == 400 || errCode == 403) {
                        showDebugToast("🔑 API Key ब्लॉक या अमान्य है ($errCode)! 🔊 बटन को दबाए रखकर नई Key डालें।")
                    }
                }
            }

            // 3. ऑनलाइन High-Quality स्ट्रीम बैकअप
            if (tryPlayOnlineNeuralMp3(text, chosenVoice.name == "Sadachbia")) return@execute

            // 4. अंतिम ऑफलाइन बैकअप
            if (isOfflineTtsReady) {
                scheduleSpeakingUnlock(4500L)
                tts?.setPitch(if (chosenVoice.name == "Sadachbia") 0.82f else 1.05f)
                tts?.speak(text, TextToSpeech.QUEUE_FLUSH, null, "SARVODAYA_FALLBACK")
            } else {
                isCurrentlySpeaking = false
            }
        }
    }

    private fun showDebugToast(msg: String) {
        mainHandler.post {
            Toast.makeText(context, msg, Toast.LENGTH_LONG).show()
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
                    connectTimeout = 6000
                    readTimeout = 8000
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
                } else if (lastHttpCode == 400 || lastHttpCode == 403) {
                    break
                }
            } catch (e: Exception) {
                e.printStackTrace()
            }
        }
        return Pair(null, lastHttpCode)
    }

    // ✅ ऑडियो सैंपल्स को डिजिटली बूस्ट (Amplify) करें ताकि आवाज़ साफ़ और तेज़ आए
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

            // हार्डवेयर LoudnessEnhancer (+600mB यानी +6dB अतिरिक्त आवाज़ बूस्ट)
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
            e.printStackTrace()
            isCurrentlySpeaking = false
            false
        }
    }

    private fun tryPlayOnlineNeuralMp3(text: String, isLowerPitchMale: Boolean): Boolean {
        return try {
            val encoded = URLEncoder.encode(text, "UTF-8")
            val urlStr = "https://translate.googleapis.com/translate_tts?ie=UTF-8&q=$encoded&tl=hi&client=tw-ob"
            val conn = (URL(urlStr).openConnection() as HttpURLConnection).apply {
                connectTimeout = 3000
                readTimeout = 4000
                setRequestProperty("User-Agent", "Mozilla/5.0")
            }
            if (conn.responseCode == 200) {
                val mp3File = File(voiceCacheDir, "temp_neural.mp3")
                conn.inputStream.use { input ->
                    FileOutputStream(mp3File).use { output ->
                        input.copyTo(output)
                    }
                }
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
                                setTargetGain(600)
                                enabled = true
                            }
                        } catch (_: Exception) {}
                        if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.M) {
                            val targetPitch = if (isLowerPitchMale) 0.85f else 1.04f
                            playbackParams = playbackParams.setPitch(targetPitch).setSpeed(0.98f)
                        }
                        val dur = duration.toLong().coerceAtLeast(3000L)
                        scheduleSpeakingUnlock(dur + 400L)
                        setOnCompletionListener {
                            isCurrentlySpeaking = false
                        }
                        start()
                    }
                }
                true
            } else false
        } catch (e: Exception) {
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
