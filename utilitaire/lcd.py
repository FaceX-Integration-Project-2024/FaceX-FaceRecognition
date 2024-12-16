from machine import Pin
import time

# Configuration des broches
rs = Pin(8, Pin.OUT)
enable = Pin(9, Pin.OUT)
d4 = Pin(10, Pin.OUT)
d5 = Pin(11, Pin.OUT)
d6 = Pin(12, Pin.OUT)
d7 = Pin(13, Pin.OUT)

# Fonction pour envoyer des données en 4 bits
def lcd_send(data, is_command):
    rs.value(0 if is_command else 1)  # Mode commande ou donnée

    # Envoi des bits hauts
    d4.value((data >> 4) & 0x01)
    d5.value((data >> 5) & 0x01)
    d6.value((data >> 6) & 0x01)
    d7.value((data >> 7) & 0x01)
    lcd_toggle_enable()

    # Envoi des bits bas
    d4.value(data & 0x01)
    d5.value((data >> 1) & 0x01)
    d6.value((data >> 2) & 0x01)
    d7.value((data >> 3) & 0x01)
    lcd_toggle_enable()

# Fonction pour activer la broche Enable
def lcd_toggle_enable():
    enable.value(1)
    time.sleep_us(1)  # Pause pour signaler la commande
    enable.value(0)
    time.sleep_us(50)  # Délai d'attente

# Initialisation de l'écran LCD
def lcd_init():
    time.sleep(0.05)

    # Initialisation en mode 4 bits
    lcd_send(0x03, True)
    lcd_send(0x03, True)
    lcd_send(0x03, True)
    lcd_send(0x02, True)  # Passer en mode 4 bits

    # Configuration de l'écran
    lcd_send(0x28, True)  # Mode 4 bits, 2 lignes, 5x8 points
    lcd_send(0x0C, True)  # Affichage activé, curseur désactivé
    lcd_send(0x06, True)  # Incrémentation automatique
    lcd_clear()

# Effacer l'écran
def lcd_clear():
    lcd_send(0x01, True)
    time.sleep_ms(2)

# Positionner le curseur
def lcd_set_cursor(line, column):
    addr = 0x80 + (0x40 * line) + column
    lcd_send(addr, True)

# Fonction pour afficher un texte
def lcd_write(message):
    for char in message:
        lcd_send(ord(char), False)

# Programme principal
lcd_init()
print("init")

# Exemple d'affichage
lcd_set_cursor(0, 0)  # Ligne 1, colonne 0
lcd_write("Hello FaceX!")
print("ecran")

lcd_set_cursor(1, 0)  # Ligne 2, colonne 0
lcd_write("Bienvenue")

