package org.sasc.sentinel

import android.content.Context

object EntropyGenerator {
    fun start(context: Context, callback: (ByteArray) -> Unit) {
        // Placeholder para loop de captura de ruído da câmera
        // Em produção, isso usaria Camera2 API ou CameraX
        val dummyNoise = "SASC_ENTROPY_DUMMY".toByteArray()
        callback(dummyNoise)
    }
}
