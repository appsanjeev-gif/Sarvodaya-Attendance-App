package com.school.sarvodayaattendance.utils

import android.content.Context
import android.media.AudioAttributes
import android.media.MediaPlayer
import android.speech.tts.TextToSpeech
import android.speech.tts.Voice
import java.io.File
import java.io.FileOutputStream
import java.net.HttpURLConnection
import java.net.URL
import java.net.URLEncoder
import java.util.Locale
import java.util.concurrent.Executors
import kotlin.random.Random

class RealisticVoiceManager(private val context: Context) : TextToSpeech.OnInitListener {

    private var tts: TextToSpeech? = null
    private var isReady = false
    var isVoiceEnabled = true
    private val netExecutor = Executors.newSingleThreadExecutor()
    private var mediaPlayer: MediaPlayer? = null

    private val successTemplates = listOf(
        "नमस्कार %s, बहुत बढ़िया! आपकी आज की उपस्थिति दर्ज हो चुकी है।",
        "स्वागत है %s! आपकी हाज़िरी सफलतापूर्वक लग गई है।",
        "शाबाश %s! आज विद्यालय में आपका स्वागत है, उपस्थिति दर्ज हो गई।",
        "धन्यवाद %s, आपकी उपस्थिति समय पर दर्ज कर ली गई है।"
    )

    private val warningTemplates = listOf(
        "ध्यान दें %s, आपकी आज की उपस्थिति पहले ही दर्ज हो चुकी है।",
        "%s, आपकी हाज़िरी आज पहले ही लग चुकी है, दोबारा आवश्यकता नहीं है।"
    )

    private val errorTemplates = listOf(
        "क्षमा करें, चेहरा स्पष्ट रूप से पहचाना नहीं जा सका, कृपया कैमरे के सामने सीधे देखें।",
        "चेहरा मिलान नहीं हुआ, कृपया रोशनी में आकर दोबारा प्रयास करें।"
    )

    init {
        tts = TextToSpeech(context, this)
    }

    override fun onInit(status: Int) {
        if (status == TextToSpeech.SUCCESS) {
            val hindiLocale = Locale("hi", "IN")
            tts?.language = hindiLocale

            // अमिताभ जी जैसी भारी और गंभीर (Deep Baritone Male) आवाज़ के लिए पिच और स्पीड सेटिंग
            tts?.setPitch(0.76f)
            tts?.setSpeechRate(0.92f)

            try {
                val voices: Set<Voice>? = tts?.voices
                val bestMaleVoice = voices?.firstOrNull { v ->
                    v.locale.language == "hi" && (v.name.contains("male", true) || v.name.contains("hid", true) || v.name.contains("hie", true) || v.name.contains("network", true))
                } ?: voices?.firstOrNull { v ->
                    v.locale.language == "hi" && !v.isNetworkConnectionRequired
                }
                if (bestMaleVoice != null) {
                    tts?.voice = bestMaleVoice
                }
            } catch (e: Exception) {
                e.printStackTrace()
            }
            isReady = true
        }
    }

    fun speakGreeting(type: String, studentName: String) {
        if (!isVoiceEnabled) return
        val cleanName = studentName.trim()
        val message = when (type) {
            "success" -> String.format(successTemplates[Random.nextInt(successTemplates.size)], cleanName)
            "warning" -> String.format(warningTemplates[Random.nextInt(warningTemplates.size)], cleanName)
            else -> errorTemplates[Random.nextInt(errorTemplates.size)]
        }
        speakHindi(message)
    }

    fun speakHindi(text: String) {
        if (!isVoiceEnabled || text.isBlank()) return

        // पहले ऑनलाइन हाई-क्वालिटी नेचुरल AI आवाज़ चलाने का प्रयास करें, इंटरनेट न होने पर तुरंत डीप-बैरीटोन ऑफलाइन इंजन चलाएं
        netExecutor.execute {
            val playedOnline = tryPlayOnlineNaturalVoice(text)
            if (!playedOnline && isReady) {
                tts?.setPitch(0.76f)
                tts?.setSpeechRate(0.92f)
                tts?.speak(text, TextToSpeech.QUEUE_FLUSH, null, "SARVODAYA_VOICE_ID")
            }
        }
    }

    private fun tryPlayOnlineNaturalVoice(text: String): Boolean {
        return try {
            val encoded = URLEncoder.encode(text, "UTF-8")
            val urlStr = "https://translate.googleapis.com/translate_tts?ie=UTF-8&q=$encoded&tl=hi&client=tw-ob&ttsspeed=0.92"
            val conn = (URL(urlStr).openConnection() as HttpURLConnection).apply {
                connectTimeout = 1800
                readTimeout = 2200
                setRequestProperty("User-Agent", "Mozilla/5.0")
            }
            if (conn.responseCode == 200) {
                val tempFile = File(context.cacheDir, "ai_voice_prompt.mp3")
                conn.inputStream.use { input ->
                    FileOutputStream(tempFile).use { output ->
                        input.copyTo(output)
                    }
                }
                mediaPlayer?.release()
                mediaPlayer = MediaPlayer().apply {
                    setAudioAttributes(
                        AudioAttributes.Builder()
                            .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH)
                            .setUsage(AudioAttributes.USAGE_ASSISTANCE_NAVIGATION_GUIDANCE)
                            .build()
                    )
                    setDataSource(tempFile.absolutePath)
                    prepare()
                    // Android PlaybackParams से आवाज़ को थोड़ा गहरा (Deep Baritone) और प्रभावशाली बनाएं
                    if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.M) {
                        playbackParams = playbackParams.setPitch(0.84f).setSpeed(0.96f)
                    }
                    start()
                }
                true
            } else {
                false
            }
        } catch (e: Exception) {
            false
        }
    }

    fun shutdown() {
        try {
            mediaPlayer?.release()
            mediaPlayer = null
            tts?.stop()
            tts?.shutdown()
        } catch (e: Exception) {
            e.printStackTrace()
        }
    }
}
