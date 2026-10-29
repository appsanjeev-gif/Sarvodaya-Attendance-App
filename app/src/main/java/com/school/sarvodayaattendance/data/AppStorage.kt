package com.school.sarvodayaattendance.data

import android.content.Context
import com.google.gson.Gson
import com.google.gson.reflect.TypeToken
import java.text.SimpleDateFormat
import java.util.Calendar
import java.util.Date
import java.util.Locale

data class Student(
    var id: String = "",
    var className: String = "6वीं",
    var roll: String = "01",
    var name: String = "",
    var gender: String = "M",
    var category: String = "EBC",
    var aadhaar: String = "-",
    var father: String = "-",
    var mother: String = "-",
    var dob: String = "-",
    var mobile: String = "-",
    var admNo: String = "-",
    var admDate: String = "-",
    var address: String = "-",
    var photos: MutableList<String> = mutableListOf()
)

data class BackupPayload(
    var students: MutableList<Student> = mutableListOf(),
    var attendance: MutableMap<String, MutableList<String>> = mutableMapOf(),
    var holidays: MutableMap<String, Any> = mutableMapOf(),
    var adminPin: String = "123456",
    var isScannerOnlyPack: Boolean = false,
    var sharedApiKeys: String = ""
)

object AppStorage {
    private const val PREF_NAME = "SarvodayaAppStorage"
    private const val KEY_PAYLOAD_JSON = "master_payload_json"
    private const val VOICE_PREF_NAME = "SarvodayaVoicePrefs"
    private const val KEY_CUSTOM_API = "custom_gemini_api_key"
    private val gson = Gson()

    fun loadData(context: Context): BackupPayload {
        val prefs = context.getSharedPreferences(PREF_NAME, Context.MODE_PRIVATE)
        val voicePrefs = context.getSharedPreferences(VOICE_PREF_NAME, Context.MODE_PRIVATE)
        val raw = prefs.getString(KEY_PAYLOAD_JSON, null)
        val payload = if (raw.isNullOrBlank()) {
            BackupPayload()
        } else {
            try {
                gson.fromJson(raw, BackupPayload::class.java) ?: BackupPayload()
            } catch (e: Exception) {
                BackupPayload()
            }
        }

        val currentVoiceKeys = voicePrefs.getString(KEY_CUSTOM_API, "") ?: ""
        if (currentVoiceKeys.isNotBlank()) {
            payload.sharedApiKeys = currentVoiceKeys
        } else if (payload.sharedApiKeys.isNotBlank()) {
            voicePrefs.edit().putString(KEY_CUSTOM_API, payload.sharedApiKeys.trim()).apply()
        }

        return payload
    }

    fun saveData(context: Context, payload: BackupPayload) {
        val voicePrefs = context.getSharedPreferences(VOICE_PREF_NAME, Context.MODE_PRIVATE)
        val currentVoiceKeys = voicePrefs.getString(KEY_CUSTOM_API, "") ?: ""
        if (currentVoiceKeys.isNotBlank()) {
            payload.sharedApiKeys = currentVoiceKeys
        }

        val prefs = context.getSharedPreferences(PREF_NAME, Context.MODE_PRIVATE)
        prefs.edit().putString(KEY_PAYLOAD_JSON, gson.toJson(payload)).apply()
    }

    fun getTodayShortKey(): String {
        return SimpleDateFormat("dd.MM.yy", Locale.US).format(Date())
    }

    fun getTodayFullDot(): String {
        return SimpleDateFormat("dd.MM.yyyy", Locale.US).format(Date())
    }

    fun formatRoll(roll: String): String {
        val clean = roll.trim()
        return if (clean.length == 1) "0$clean" else clean
    }

    fun isSundayOrHoliday(payload: BackupPayload, shortDateKey: String, targetClass: String): Boolean {
        try {
            val parts = shortDateKey.split(".")
            if (parts.size == 3) {
                val dd = parts[0].toInt()
                val mm = parts[1].toInt() - 1
                val yyyy = if (parts[2].length == 2) ("20" + parts[2]).toInt() else parts[2].toInt()
                val cal = Calendar.getInstance().apply { set(yyyy, mm, dd) }
                if (cal.get(Calendar.DAY_OF_WEEK) == Calendar.SUNDAY) return true
            }
        } catch (_: Exception) {}

        val entry = payload.holidays[shortDateKey] ?: return false
        if (entry is String) return true
        if (entry is Map<*, *>) {
            if (entry.containsKey("सभी")) return true
            if (targetClass != "सभी" && entry.containsKey(targetClass)) return true
        }
        return false
    }

    fun importFromHtmlOrJson(context: Context, rawContent: String): Boolean {
        return try {
            val trimmed = rawContent.trim()
            val jsonCandidate = if (trimmed.startsWith("{")) {
                trimmed
            } else {
                val startIdx = trimmed.indexOf("{\"students\"")
                val endIdx = trimmed.lastIndexOf("}")
                if (startIdx != -1 && endIdx != -1 && endIdx > startIdx) {
                    trimmed.substring(startIdx, endIdx + 1)
                } else {
                    return false
                }
            }

            val parsed = gson.fromJson(jsonCandidate, BackupPayload::class.java)
            if (parsed != null && parsed.students != null) {
                // ✅ यदि पैक के साथ गुप्त API Keys आई हैं, तो उन्हें इस मोबाइल के वॉइस मैनेजर में चुपचाप सेव कर दें
                if (parsed.sharedApiKeys.isNotBlank()) {
                    val voicePrefs = context.getSharedPreferences(VOICE_PREF_NAME, Context.MODE_PRIVATE)
                    voicePrefs.edit().putString(KEY_CUSTOM_API, parsed.sharedApiKeys.trim()).apply()
                }
                saveData(context, parsed)
                true
            } else {
                false
            }
        } catch (e: Exception) {
            false
        }
    }
}
