package org.sasc.sentinel

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import java.io.File
import org.sasc.sentinel.tor.TorService

class SentinelActivity : AppCompatActivity() {

    // Endereço Onion do Mobile-Hub (Exemplo)
    private val MOBILE_HUB_ONION = "sascghostfleet...xyz.onion:30303"
    private lateinit var statusText: TextView

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_sentinel)
        statusText = findViewById(R.id.statusText)

        // 1. Verificar Permissões de Câmera
        if (checkSelfPermission(Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this, arrayOf(Manifest.permission.CAMERA), 101)
        } else {
            startSentinelCore()
        }
    }

    private fun startSentinelCore() {
        // 2. Iniciar Daemon Tor (Em segundo plano)
        TorService.start(applicationContext, File(filesDir, "tor_data"))

        // 3. Iniciar Geração de Entropia (Loop)
        EntropyGenerator.start(this) { rawNoise ->
            val hash = generateEntropy(rawNoise)

            // 4. Enviar "Heartbeat" para o Hub via Tor (Encapsulado)
            NetworkClient.sendTorPost(
                MOBILE_HUB_ONION,
                "/api/v1/heartbeat",
                mapOf("entropy_source" to "CAMERA_NOISE", "hash" to hash)
            )

            runOnUiThread {
                updateUI("Status: SYNCED | Φ: Mobile-Active")
            }
        }
    }

    private fun updateUI(status: String) {
        statusText.text = status
    }

    // Nativo (JNI Bridge)
    private external fun generateEntropy(cameraBuffer: ByteArray): String
    private external fun signProof(privateKeyHex: String, message: String): String

    companion object {
        // Carregar biblioteca nativa Rust (.so)
        init {
            System.loadLibrary("sasc_core")
        }
    }
}
