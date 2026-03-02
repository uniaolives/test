package org.sasc.sentinel.tor

import android.app.Service
import android.content.Context
import android.content.Intent
import android.os.IBinder
import java.io.File

class TorService : Service() {

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        val dataDir = File(filesDir, "tor_data")
        println("TorService: Starting daemon in ${dataDir.absolutePath}")
        return START_STICKY
    }

    companion object {
        fun start(context: Context, dataDir: File) {
            val intent = Intent(context, TorService::class.java)
            context.startService(intent)
        }
    }
}
