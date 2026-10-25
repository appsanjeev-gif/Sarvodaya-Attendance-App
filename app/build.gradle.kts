plugins {
    id 'com.android.application'
    id 'org.jetbrains.kotlin.android'
}

android {
    namespace 'com.school.sarvodayaattendance'
    compileSdk 34

    defaultConfig {
        applicationId "com.school.sarvodayaattendance"
        minSdk 24
        targetSdk 34
        versionCode 16
        versionName "16.0-AI"
    }

    // AI मॉडल (.tflite) को कंप्रेस होने से बचाएं ताकि वह तेज़ चले
    aaptOptions {
        noCompress "tflite"
    }

    buildTypes {
        release {
            minifyEnabled false
            proguardFiles getDefaultProguardFile('proguard-android-optimize.txt'), 'proguard-rules.pro'
        }
    }
    compileOptions {
        sourceCompatibility JavaVersion.VERSION_17
        targetCompatibility JavaVersion.VERSION_17
    }
    kotlinOptions {
        jvmTarget = '17'
    }
}

dependencies {
    implementation 'androidx.core:core-ktx:1.12.0'
    implementation 'androidx.appcompat:appcompat:1.6.1'
    implementation 'com.google.android.material:material:1.11.0'
    implementation 'androidx.constraintlayout:constraintlayout:2.1.4'

    def camerax_version = "1.3.1"
    implementation "androidx.camera:camera-core:${camerax_version}"
    implementation "androidx.camera:camera-camera2:${camerax_version}"
    implementation "androidx.camera:camera-lifecycle:${camerax_version}"
    implementation "androidx.camera:camera-view:${camerax_version}"

    // Google ML Kit (चेहरा ढूँढने के लिए)
    implementation 'com.google.mlkit:face-detection:16.1.6'
    implementation 'com.google.android.gms:play-services-tasks:18.1.0'

    // TensorFlow Lite (मोबाइल फेस-लॉक जैसा असली Deep Learning Face Recognition करने के लिए)
    implementation 'org.tensorflow:tensorflow-lite:2.14.0'

    implementation 'com.google.code.gson:gson:2.10.1'
}
