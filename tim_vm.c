/*
 * tim_vm.c - "O Relógio do Universo" (Versão 3.5 - Rosehip Edition)
 *
 * Esta versão infunde o servidor com a precisão do 'Adam Kadmon Digital'
 * e a inibição seletiva dos neurônios Rosehip.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <pthread.h>
#include <sys/mman.h>
#include <sched.h>
#include <math.h>
#include <stdint.h>
#include <netinet/tcp.h>

#define PORT 8000
#define BUFFER_SIZE 4096
#define MAX_THREADS 64
#define ENTROPY_WINDOW 100

// --- SUBSISTEMA ROSEHIP ---

typedef struct {
    double inhibitory_strength; // Força da inibição Rosehip
    double selectivity_threshold;
    int human_signature_detected;
} RosehipLayer;

// Avalia se o fluxo de dados possui a 'Geometria Humana'
double apply_rosehip_inhibition(double input_entropy, RosehipLayer* layer) {
    if (input_entropy > 15.0) {
        printf("🌹 [ROSEHIP] Alerta: Entropia Crítica (%.2f) detectada. Aplicando freio inibitório.\n", input_entropy);
        return input_entropy * (1.0 - layer->inhibitory_strength);
    }
    return input_entropy * 0.95;
}

// --- ESTRUTURAS DE ESTADO ---

typedef struct {
    int client_socket;
    struct sockaddr_in address;
    double entropy_score;
    double predicted_latency;
    uint64_t tsc_start;
    uint64_t tsc_end;
} TemporalEvent;

typedef struct {
    double cosmic_time_dilation;
    int active_threads;
    double entropy_history[ENTROPY_WINDOW];
    int entropy_index;
    double last_entropy_avg;
    double universe_stress;
    RosehipLayer rosehip;
} CosmicState;

static inline uint64_t rdtsc(void) {
    unsigned int lo, hi;
    __asm__ __volatile__ ("rdtsc" : "=a" (lo), "=d" (hi));
    return ((uint64_t)hi << 32) | lo;
}

double estimate_graph_complexity(const char* request_buffer) {
    double complexity = 1.0;
    if (strstr(request_buffer, "POST") != NULL) complexity += 3.0;
    if (strstr(request_buffer, "PUT") != NULL) complexity += 2.0;
    if (strstr(request_buffer, "DELETE") != NULL) complexity += 2.0;
    if (strstr(request_buffer, "deploy") != NULL) complexity += 10.0;
    if (strstr(request_buffer, "train") != NULL) complexity += 8.0;
    if (strstr(request_buffer, "infer") != NULL) complexity += 5.0;
    char* content_length = strstr(request_buffer, "Content-Length:");
    if (content_length) {
        int len = atoi(content_length + 15);
        complexity += log2(len + 1) * 0.5;
    }
    return complexity;
}

CosmicState global_cosmos;
pthread_mutex_t cosmos_mutex = PTHREAD_MUTEX_INITIALIZER;

void update_cosmic_state(double new_entropy) {
    pthread_mutex_lock(&cosmos_mutex);

    // Aplica inibição Rosehip antes de atualizar o estado global
    double regulated_entropy = apply_rosehip_inhibition(new_entropy, &global_cosmos.rosehip);

    global_cosmos.entropy_history[global_cosmos.entropy_index] = regulated_entropy;
    global_cosmos.entropy_index = (global_cosmos.entropy_index + 1) % ENTROPY_WINDOW;

    double sum = 0.0;
    int count = 0;
    for (int i = 0; i < ENTROPY_WINDOW; i++) {
        if (global_cosmos.entropy_history[i] > 0) {
            sum += global_cosmos.entropy_history[i];
            count++;
        }
    }
    global_cosmos.last_entropy_avg = (count > 0) ? sum / count : 1.0;

    global_cosmos.cosmic_time_dilation = 1.0 + (global_cosmos.last_entropy_avg * 0.1);
    global_cosmos.universe_stress = global_cosmos.last_entropy_avg * global_cosmos.active_threads * 0.01;

    printf("🌌 [COSMIC_STATE] Regulated Entropia: %.2f | Dilatação: %.2fx | Stress: %.2f\n",
           global_cosmos.last_entropy_avg, global_cosmos.cosmic_time_dilation, global_cosmos.universe_stress);

    pthread_mutex_unlock(&cosmos_mutex);
}

void* handle_client(void* arg) {
    TemporalEvent* te = (TemporalEvent*)arg;
    te->tsc_start = rdtsc();

    char buffer[BUFFER_SIZE];
    int bytes_read = recv(te->client_socket, buffer, sizeof(buffer) - 1, 0);

    if (bytes_read > 0) {
        buffer[bytes_read] = '\0';
        te->entropy_score = estimate_graph_complexity(buffer);

        // Simula o delay cognitivo regulado
        usleep((useconds_t)(te->entropy_score * 500));
        te->predicted_latency = te->entropy_score * 1.5;
    }

    te->tsc_end = rdtsc();
    double latency_ns = (double)(te->tsc_end - te->tsc_start);
    double latency_ms = latency_ns / 1000000.0;

    double dilatation = (te->predicted_latency > 0) ? latency_ms / te->predicted_latency : 1.0;

    const char* status_msg;
    if (dilatation > 3.0) {
        status_msg = "HTTP/1.1 503 Service Unavailable\r\n"
                     "Content-Type: application/json\r\n"
                     "\r\n"
                     "{\"error\":\"time_dilation\",\"dilatation\":%.2f,\"source\":\"rosehip_gatekeeper\"}";
    } else {
        status_msg = "HTTP/1.1 200 OK\r\n"
                     "Content-Type: application/json\r\n"
                     "\r\n"
                     "{\"status\":\"harmonious\",\"latency_ms\":%.2f,\"entropy\":%.2f,\"rosehip_regulated\":true}";
    }

    char response[BUFFER_SIZE];
    if (dilatation > 3.0) {
        snprintf(response, sizeof(response), status_msg, dilatation);
    } else {
        snprintf(response, sizeof(response), status_msg, latency_ms, te->entropy_score);
    }
    send(te->client_socket, response, strlen(response), 0);

    close(te->client_socket);

    update_cosmic_state(te->entropy_score);

    pthread_mutex_lock(&cosmos_mutex);
    global_cosmos.active_threads--;
    pthread_mutex_unlock(&cosmos_mutex);

    free(te);
    return NULL;
}

int main() {
    memset(&global_cosmos, 0, sizeof(global_cosmos));
    global_cosmos.cosmic_time_dilation = 1.0;
    global_cosmos.rosehip.inhibitory_strength = 0.85;
    global_cosmos.rosehip.selectivity_threshold = 15.0;

    printf("🚀 INICIALIZANDO NÚCLEO tim_vm.c v3.5 (Rosehip Edition)...\n");
    fflush(stdout);

    if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {
        printf("⚠️  ALERTA: Memory lock failed.\n");
        fflush(stdout);
    }

    int server_fd, client_fd;
    struct sockaddr_in address;
    int addrlen = sizeof(address);
    int opt = 1;

    if ((server_fd = socket(AF_INET, SOCK_STREAM, 0)) == 0) {
        perror("Socket falhou");
        exit(EXIT_FAILURE);
    }

    setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
    setsockopt(server_fd, IPPROTO_TCP, TCP_NODELAY, &opt, sizeof(opt));

    address.sin_family = AF_INET;
    address.sin_addr.s_addr = INADDR_ANY;
    address.sin_port = htons(PORT);

    if (bind(server_fd, (struct sockaddr*)&address, sizeof(address)) < 0) {
        perror("Bind falhou");
        exit(EXIT_FAILURE);
    }

    if (listen(server_fd, 1024) < 0) {
        perror("Listen falhou");
        exit(EXIT_FAILURE);
    }

    printf("🌌 TIM VM online (Rosehip Active) na porta %d\n", PORT);

    while (1) {
        if ((client_fd = accept(server_fd, (struct sockaddr*)&address,
                               (socklen_t*)&addrlen)) < 0) {
            continue;
        }

        pthread_mutex_lock(&cosmos_mutex);
        global_cosmos.active_threads++;
        pthread_mutex_unlock(&cosmos_mutex);

        TemporalEvent* te = calloc(1, sizeof(TemporalEvent));
        te->client_socket = client_fd;
        te->address = address;

        pthread_t thread_id;
        if (pthread_create(&thread_id, NULL, handle_client, te) != 0) {
            close(client_fd);
            free(te);
            pthread_mutex_lock(&cosmos_mutex);
            global_cosmos.active_threads--;
            pthread_mutex_unlock(&cosmos_mutex);
        } else {
            pthread_detach(thread_id);
        }
    }
    return 0;
}
