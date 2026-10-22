package com.school.sarvodayaattendance.utils

import android.content.Context
import android.content.Intent
import android.media.AudioManager
import android.media.ToneGenerator
import android.speech.tts.TextToSpeech
import android.speech.tts.Voice
import java.util.Locale
import kotlin.random.Random

class RealisticVoiceManager(private val context: Context) : TextToSpeech.OnInitListener {

    private var tts: TextToSpeech? = null
    private var isReady = false
    var isVoiceEnabled = true
    private val hindiVoices = mutableListOf<Voice>()
    private val englishVoices = mutableListOf<Voice>()
    private var toneGen: ToneGenerator? = null

    init {
        try {
            toneGen = ToneGenerator(AudioManager.STREAM_MUSIC, 90)
        } catch (e: Exception) {
            e.printStackTrace()
        }
        tts = TextToSpeech(context.applicationContext, this, "com.google.android.tts")
    }

    override fun onInit(status: Int) {
        if (status == TextToSpeech.SUCCESS) {
            isReady = true
            refreshAvailableVoices()
        }
    }

    fun refreshAvailableVoices() {
        val engine = tts ?: return
        try {
            val allVoices = engine.voices ?: return
            hindiVoices.clear()
            englishVoices.clear()
            for (v in allVoices) {
                val lang = v.locale.language.lowercase()
                val country = v.locale.country.uppercase()
                if (lang == "hi") {
                    hindiVoices.add(v)
                } else if (lang == "en" && country == "IN") {
                    englishVoices.add(v)
                }
            }
            val res = engine.isLanguageAvailable(Locale("hi", "IN"))
            if (res == TextToSpeech.LANG_MISSING_DATA) {
                promptInstallHighQualityVoices()
            }
        } catch (e: Exception) {
            e.printStackTrace()
        }
    }

    fun promptInstallHighQualityVoices() {
        try {
            val intent = Intent(TextToSpeech.Engine.ACTION_INSTALL_TTS_DATA)
            intent.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
            context.startActivity(intent)
        } catch (e: Exception) {
            e.printStackTrace()
        }
    }

    fun playBeep(type: String) {
        try {
            when (type) {
                "success" -> toneGen?.startTone(ToneGenerator.TONE_PROP_ACK, 180)
                "warning" -> toneGen?.startTone(ToneGenerator.TONE_PROP_BEEP2, 220)
                else -> toneGen?.startTone(ToneGenerator.TONE_PROP_NACK, 280)
            }
        } catch (e: Exception) {
            e.printStackTrace()
        }
    }

    fun speakGreeting(type: String, studentName: String) {
        playBeep(type)
        if (!isVoiceEnabled || !isReady) return
        val engine = tts ?: return

        val useEnglish = Random.nextDouble() < 0.35
        val text = when (type) {
            "success" -> {
                if (useEnglish) {
                    listOf(
                        "Well done $studentName, attendance marked!",
                        "Welcome $studentName, have a great day!",
                        "Good job $studentName, you are present today!"
                    ).random()
                } else {
                    listOf(
                        "शाबाश $studentName, आपकी उपस्थिति दर्ज हो गई!",
                        "स्वागत है $studentName, आज का दिन शुभ हो!",
                        "बहुत बढ़िया $studentName, हाज़िरी लग गई!",
                        "सुप्रभात $studentName, आप उपस्थित हैं!"
                    ).random()
                }
            }
            "warning" -> {
                if (useEnglish) {
                    "Attention $studentName, attendance is already marked!"
                } else {
                    "$studentName, आपकी हाज़िरी आज पहले ही लग चुकी है!"
                }
            }
            else -> {
                if (useEnglish) {
                    "Face not recognized, please look at the camera."
                } else {
                    "चेहरा मैच नहीं हुआ, कृपया कैमरे की ओर देखें।"
                }
            }
        }

        try {
            if (useEnglish && englishVoices.isNotEmpty()) {
                engine.voice = englishVoices.random()
            } else if (!useEnglish && hindiVoices.isNotEmpty()) {
                engine.voice = hindiVoices.random()
            } else {
                engine.language = if (useEnglish) Locale("en", "IN") else Locale("hi", "IN")
            }
            engine.setPitch(Random.nextDouble(0.96, 1.06).toFloat())
            engine.setSpeechRate(0.98f)
            engine.speak(text, TextToSpeech.QUEUE_FLUSH, null, "UTTER_${System.currentTimeMillis()}")
        } catch (e: Exception) {
            e.printStackTrace()
        }
    }

    fun shutdown() {
        try {
            tts?.stop()
            tts?.shutdown()
            toneGen?.release()
        } catch (e: Exception) {
            e.printStackTrace()
        }
    }
}
