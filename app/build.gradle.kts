plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "com.school.sarvodayaattendance"
    compileSdk = 34

    defaultConfig {
        applicationId = "com.school.sarvodayaattendance"
        minSdk = 24
        targetSdk = 34
        versionCode = 16
        versionName = "16.0-AI"
    }

    // AI मॉडल (.tflite) को कंप्रेस होने से बचाएं ताकि वह तेज़ चले
    androidResources {
        noCompress += "tflite"
    }

    buildTypes {
        release {
            isMinifyEnabled = false
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_11
        targetCompatibility = JavaVersion.VERSION_11
    }

    kotlinOptions {
        jvmTarget = "11"
    }
}

dependencies {
    implementation("androidx.core:core-ktx:1.12.0")
    implementation("androidx.appcompat:appcompat:1.6.1")
    implementation("com.google.android.material:material:1.11.0")
    implementation("androidx.constraintlayout:constraintlayout:2.1.4")
    implementation("androidx.recyclerview:recyclerview:1.3.2")
    implementation("androidx.activity:activity-ktx:1.8.2")

    // CameraX & Guava
    implementation("androidx.camera:camera-core:1.3.1")
    implementation("androidx.camera:camera-camera2:1.3.1")
    implementation("androidx.camera:camera-lifecycle:1.3.1")
    implementation("androidx.camera:camera-view:1.3.1")
    implementation("com.google.guava:guava:31.1-android")

    // Google ML Kit Face Detection
    implementation("com.google.mlkit:face-detection:16.1.6")
    implementation("com.google.android.gms:play-services-tasks:18.1.0")

    // TensorFlow Lite (मोबाइल फेस-लॉक जैसा असली Deep Learning Face Recognition करने के लिए)
    implementation("org.tensorflow:tensorflow-lite:2.14.0")

    // Gson JSON Parser & Coroutines
    implementation("com.google.code.gson:gson:2.10.1")
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.7.3")
}
