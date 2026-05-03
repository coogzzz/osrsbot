#include <Mouse.h>
#include <Keyboard.h>

// ── Protocol constants ──────────────────────────────────────────────────────
#define PACKET_SIZE   8

#define CMD_MOVE      0x01
#define CMD_CLICK     0x02
#define CMD_PRESS     0x03
#define CMD_REL       0x04
#define CMD_PING      0x05
#define CMD_TYPE      0x07

#define RESP_ACK      0x06
#define RESP_NAK      0x15
#define RESP_READY    0x11

// ── Packet buffer ───────────────────────────────────────────────────────────
static uint8_t pktBuf[PACKET_SIZE];
static uint8_t pktIdx = 0;

// ── Helpers ─────────────────────────────────────────────────────────────────
static uint8_t computeChecksum(const uint8_t* data, uint8_t len) {
    uint8_t sum = 0;
    for (uint8_t i = 0; i < len; i++) {
        sum += data[i];
    }
    return sum & 0xFF;
}

static void sendByte(uint8_t b) {
    Serial.write(b);
    Serial.flush();
}

static void moveRelativeClamped(int16_t dx, int16_t dy) {
    while (dx != 0 || dy != 0) {
        int8_t mx = constrain(dx, -127, 127);
        int8_t my = constrain(dy, -127, 127);
        Mouse.move(mx, my, 0);
        dx -= mx;
        dy -= my;
        delayMicroseconds(500);
    }
}

// ── Command handlers ────────────────────────────────────────────────────────
static void handleMove(int16_t x, int16_t y) {
    moveRelativeClamped(x, y);
}

static void handleClick(uint16_t durationMs) {
    if (durationMs < 10)  durationMs = 10;
    Mouse.press(MOUSE_LEFT);
    delay(durationMs);
    Mouse.release(MOUSE_LEFT);
}

static void handleType(uint8_t key, uint16_t durationMs) {
    // If duration is 0, we treat it as a standard tap
    if (durationMs == 0) {
        Keyboard.write(key);
    } else {
        // Hold the key (essential for OSRS camera rotation)
        Keyboard.press(key);
        delay(durationMs);
        Keyboard.release(key);
    }
}

// ── Packet processor ────────────────────────────────────────────────────────
static void processPacket() {
    uint8_t expected = computeChecksum(pktBuf, PACKET_SIZE - 1);
    if (pktBuf[PACKET_SIZE - 1] != expected) {
        sendByte(RESP_NAK);
        return;
    }

    uint8_t cmd = pktBuf[0];
    int16_t  x   = (int16_t)(pktBuf[1] | (pktBuf[2] << 8));
    int16_t  y   = (int16_t)(pktBuf[3] | (pktBuf[4] << 8));
    uint16_t dur = (uint16_t)(pktBuf[5] | (pktBuf[6] << 8));

    sendByte(RESP_ACK);

    switch (cmd) {
        case CMD_MOVE:  handleMove(x, y); break;
        case CMD_CLICK: handleClick(dur); break;
        case CMD_PRESS: Mouse.press(MOUSE_LEFT); break;
        case CMD_REL:   Mouse.release(MOUSE_LEFT); break;
        case CMD_TYPE:  handleType((uint8_t)x, dur); break; // X is the key code
        case CMD_PING:  break;
    }

    sendByte(RESP_READY);
}

void setup() {
    Serial.begin(115200);
    Mouse.begin();
    Keyboard.begin();
    pinMode(LED_BUILTIN, OUTPUT);
}

void loop() {
    while (Serial.available() > 0) {
        pktBuf[pktIdx++] = Serial.read();
        if (pktIdx >= PACKET_SIZE) {
            processPacket();
            pktIdx = 0;
        }
    }
}