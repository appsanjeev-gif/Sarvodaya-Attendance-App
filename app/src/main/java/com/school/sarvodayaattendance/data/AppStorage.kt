package com.school.sarvodayaattendance.data

import android.content.Context
import com.google.gson.Gson
import com.google.gson.reflect.TypeToken
import java.text.SimpleDateFormat
import java.util.Calendar
import java.util.Date
import java.util.Locale

data class Student(
    val id: String,
    var className: String,
    var roll: String,
    var name: String,
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
    var photos: MutableList<String> = mutableListOf(),
    var faceEmbedding: FloatArray? = null
)

data class BackupPayload(
    var students: MutableList<Student> = mutableListOf(),
    var attendance: MutableMap<String, MutableList<String>> = mutableMapOf(),
    var holidays: MutableMap<String, MutableMap<String, String>> = mutableMapOf(),
    var adminPin: String = "123456",
    var isScannerOnlyPack: Boolean = false
)

object AppStorage {
    private const val PREF_NAME = "sarvodaya_native_db_v16"
    private const val KEY_PAYLOAD = "master_payload_json"
    private val gson = Gson()

    fun loadData(context: Context): BackupPayload {
        val prefs = context.getSharedPreferences(PREF_NAME, Context.MODE_PRIVATE)
        val raw = prefs.getString(KEY_PAYLOAD, null) ?: return BackupPayload()
        return try {
            gson.fromJson(raw, BackupPayload::class.java) ?: BackupPayload()
        } catch (e: Exception) {
            BackupPayload()
        }
    }

    fun saveData(context: Context, payload: BackupPayload) {
        val prefs = context.getSharedPreferences(PREF_NAME, Context.MODE_PRIVATE)
        prefs.edit().putString(KEY_PAYLOAD, gson.toJson(payload)).apply()
    }

    fun formatRoll(rollStr: String): String {
        val clean = rollStr.trim()
        return if (clean.length == 1) "0$clean" else clean
    }

    fun getTodayShortKey(): String {
        return SimpleDateFormat("dd.MM.yy", Locale.US).format(Date())
    }

    fun getTodayFullDot(): String {
        return SimpleDateFormat("dd.MM.yyyy", Locale.US).format(Date())
    }

    fun isSundayOrHoliday(payload: BackupPayload, dateKey: String, targetClass: String): Boolean {
        return getHolidayReason(payload, dateKey, targetClass) != null
    }

    fun getHolidayReason(payload: BackupPayload, dateKey: String, targetClass: String): String? {
        val holMap = payload.holidays[dateKey]
        if (holMap != null) {
            if (holMap.containsKey(targetClass)) return "छुट्टी - ${holMap[targetClass]}"
            if (holMap.containsKey("सभी")) return "छुट्टी - ${holMap["सभी"]}"
        }
        return try {
            val parts = dateKey.split(".")
            if (parts.size == 3) {
                val cal = Calendar.getInstance()
                val yyyy = if (parts[2].length == 2) ("20" + parts[2]).toInt() else parts[2].toInt()
                cal.set(yyyy, parts[1].toInt() - 1, parts[0].toInt())
                if (cal.get(Calendar.DAY_OF_WEEK) == Calendar.SUNDAY) "रविवार अवकाश" else null
            } else null
        } catch (e: Exception) {
            null
        }
    }

    // HTML (v15/v16) या JSON बैकअप फ़ाइल को 100% सटीक रूप से निकालने और सेव करने का इंजन
    fun importFromHtmlOrJson(context: Context, fileContent: String): Boolean {
        return try {
            var jsonStr = fileContent.trim()
            if (!jsonStr.startsWith("{")) {
                val regex = Regex("""<script[^>]*id=["']embeddedAppData["'][^>]*>([\s\S]*?)</script>""", RegexOption.IGNORE_CASE)
                val match = regex.find(jsonStr)
                if (match != null) {
                    jsonStr = match.groupValues[1].trim()
                } else {
                    val idx = jsonStr.indexOf("embeddedAppData")
                    if (idx != -1) {
                        val start = jsonStr.indexOf(">", idx)
                        val end = jsonStr.indexOf("</script>", start)
                        if (start != -1 && end != -1) {
                            jsonStr = jsonStr.substring(start + 1, end).trim()
                        }
                    }
                }
            }

            val mapType = object : TypeToken<Map<String, Any>>() {}.type
            val rawMap: Map<String, Any> = gson.fromJson(jsonStr, mapType)

            val studentsJson = gson.toJson(rawMap["students"] ?: emptyList<Any>())
            val stuListType = object : TypeToken<MutableList<Student>>() {}.type
            val stuList: MutableList<Student> = gson.fromJson(studentsJson, stuListType)

            val attJson = gson.toJson(rawMap["attendance"] ?: emptyMap<String, Any>())
            val attType = object : TypeToken<MutableMap<String, MutableList<String>>>() {}.type
            val attMap: MutableMap<String, MutableList<String>> = gson.fromJson(attJson, attType)

            val normalizedHolidays = mutableMapOf<String, MutableMap<String, String>>()
            val rawHolidays = rawMap["holidays"]
            if (rawHolidays is Map<*, *>) {
                for ((k, v) in rawHolidays) {
                    val dateKey = k?.toString() ?: continue
                    if (v is String) {
                        normalizedHolidays[dateKey] = mutableMapOf("सभी" to v)
                    } else if (v is Map<*, *>) {
                        val scopeMap = mutableMapOf<String, String>()
                        for ((sk, sv) in v) {
                            if (sk != null && sv != null) {
                                scopeMap[sk.toString()] = sv.toString()
                            }
                        }
                        normalizedHolidays[dateKey] = scopeMap
                    }
                }
            }

            val current = loadData(context)
            current.students = stuList
            current.attendance = attMap
            current.holidays = normalizedHolidays
            if (rawMap["adminPin"] is String) {
                current.adminPin = rawMap["adminPin"] as String
            }
            current.isScannerOnlyPack = (rawMap["isScannerOnlyPack"] as? Boolean) ?: false
            saveData(context, current)
            true
        } catch (e: Exception) {
            e.printStackTrace()
            false
        }
    }
}
