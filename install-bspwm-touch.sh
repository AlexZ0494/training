#!/bin/bash
# ============================================================
# install-bspwm-touch.sh
# Автоматическая установка bspwm + polybar + rofi + тач-управление
# Для Arch Linux (ядро LTS, без графического интерфейса)
# ============================================================
set -euo pipefail

# --- Цвета ---
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

info()    { echo -e "${BLUE}[INFO]${NC}  $*"; }
success() { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error()   { echo -e "${RED}[ERR]${NC}  $*"; exit 1; }

# --- Проверки ---
[[ $EUID -eq 0 ]] && error "Не запускай от root! Скрипт использует sudo где нужно."
source /etc/os-release
[[ "$ID" != "arch" ]] && error "Это не Arch Linux. Скрипт только для Arch."

if ! command -v sudo &>/dev/null; then
    error "sudo не установлен. Установи: pacman -S sudo"
fi

GESTURE_TOOL=""        # libinput-gestures | fusuma | none
INSTALL_KEYBOARD="yes" # onboard
TERMINAL=""            # alacritty | kitty
GESTURE_DAEMON=""      # что запускать в bspwmrc

# --- Аргументы ---
usage() {
    cat <<EOF
Использование: $0 [ОПЦИИ]

Опции:
  -g, --gestures TOOL    Выбрать инструмент жестов:
                          libinput-gestures (по умолчанию)
                          fusuma
                          none (не устанавливать)
  -t, --terminal TERM   Терминал: alacritty (по умолчанию) или kitty
  -k, --no-keyboard     Не ставить экранную клавиатуру (onboard)
  -h, --help            Эта справка

Примеры:
  $0
  $0 --gestures fusuma --terminal kitty
  $0 --no-keyboard
EOF
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -g|--gestures)    GESTURE_TOOL="$2"; shift 2 ;;
        -t|--terminal)    TERMINAL="$2"; shift 2 ;;
        -k|--no-keyboard) INSTALL_KEYBOARD="no"; shift ;;
        -h|--help)        usage ;;
        *) error "Неизвестная опция: $1. Запусти с --help" ;;
    esac
done

[[ -z "$GESTURE_TOOL" ]] && GESTURE_TOOL="libinput-gestures"
[[ -z "$TERMINAL" ]]    && TERMINAL="alacritty"

# --- Меню выбора (если аргументы не переданы) ---
if [[ -t 0 ]]; then
    if [[ "$GESTURE_TOOL" == "libinput-gestures" && $# -eq 0 ]]; then
        echo ""
        echo "Выбери инструмент для тач-жестов:"
        echo "  1) libinput-gestures  (Python, стабильный, по умолчанию)"
        echo "  2) fusuma             (Ruby, больше жестов: rotate, hold)"
        echo "  3) Без жестов"
        echo ""
        read -rp "Ваш выбор [1]: " choice
        case "$choice" in
            2) GESTURE_TOOL="fusuma" ;;
            3) GESTURE_TOOL="none" ;;
            *) GESTURE_TOOL="libinput-gestures" ;;
        esac
    fi
fi

echo ""
echo "=========================================="
echo " Установка bspwm + polybar + rofi + тач"
echo "=========================================="
echo " ОС:              Arch Linux"
echo " Терминал:        $TERMINAL"
echo " Жесты:           $GESTURE_TOOL"
echo " Экран. клавиат.: $INSTALL_KEYBOARD"
echo "=========================================="
echo ""
read -rp "Продолжить? [Y/n]: " confirm
[[ "${confirm,,}" == "n" ]] && exit 0

# --- pacman пакеты ---
PACMAN_PACKAGES=(
    xorg-server xorg-xinit xorg-apps
    xf86-input-libinput xdotool wmctrl
    bspwm sxhkd polybar rofi
    picom ttf-jetbrains-mono-nerd noto-fonts-emoji
    git
)

case "$TERMINAL" in
    alacritty) PACMAN_PACKAGES+=(alacritty) ;;
    kitty)     PACMAN_PACKAGES+=(kitty) ;;
    *)         warn "Неизвестный терминал '$TERMINAL', ставлю alacritty"; TERMINAL="alacritty"; PACMAN_PACKAGES+=(alacritty) ;;
esac

[[ "$INSTALL_KEYBOARD" == "yes" ]] && PACMAN_PACKAGES+=(onboard)

# ============================================================
# Шаг 1. Обновление и установка пакетов из pacman
# ============================================================
info "Обновление системы..."
sudo pacman -Syu --noconfirm

info "Установка пакетов из официальных репозиториев..."
sudo pacman -S --noconfirm --needed "${PACMAN_PACKAGES[@]}"
success "Пакеты из pacman установлены"

# ============================================================
# Шаг 2. Установка AUR-хелпера (yay)
# ============================================================
if ! command -v yay &>/dev/null; then
    info "Установка yay (AUR helper)..."
    TMPDIR_YAY="/tmp/yay-build-$$"
    git clone https://aur.archlinux.org/yay.git "$TMPDIR_YAY"
    (cd "$TMPDIR_YAY" && makepkg -si --noconfirm)
    rm -rf "$TMPDIR_YAY"
    success "yay установлен"
else
    success "yay уже установлен"
fi

# ============================================================
# Шаг 3. Установка инструмента жестов
# ============================================================
if [[ "$GESTURE_TOOL" == "libinput-gestures" ]]; then
    info "Установка libinput-gestures из AUR..."
    yay -S --noconfirm --needed libinput-gestures
    success "libinput-gestures установлен"
    GESTURE_DAEMON="libinput-gestures-setup start"
elif [[ "$GESTURE_TOOL" == "fusuma" ]]; then
    info "Установка fusuma..."
    sudo pacman -S --noconfirm --needed ruby libinput
    sudo gem install fusuma
    sudo gem install fusuma-plugin-sendkey
    success "fusuma установлен"
    GESTURE_DAEMON="fusuma -d"
elif [[ "$GESTURE_TOOL" == "none" ]]; then
    warn "Жесты не устанавливаются"
    GESTURE_DAEMON=""
fi

# ============================================================
# Шаг 4. Права на тач-устройства
# ============================================================
info "Настройка прав доступа к тач-устройствам..."
if ! groups "$USER" | grep -qw input; then
    sudo gpasswd -a "$USER" input
    success "Пользователь $USER добавлен в группу input"
else
    success "Пользователь уже в группе input"
fi

# udev rule для тачскрина (безопасный доступ через uaccess)
UDEV_FILE="/etc/udev/rules.d/71-touch-input.rules"
if [[ ! -f "$UDEV_FILE" ]]; then
    sudo tee "$UDEV_FILE" > /dev/null << 'UDEV'
ACTION!="remove", ENV{ID_INPUT_TOUCHPAD}=="1", TAG+="uaccess"
ACTION!="remove", ENV{ID_INPUT_TOUCHSCREEN}=="1", TAG+="uaccess"
UDEV
    success "udev правило создано: $UDEV_FILE"
else
    success "udev правило уже существует"
fi

# ============================================================
# Шаг 5. Конфиг Xorg для тача
# ============================================================
info "Настройка Xorg для тачскрина/тачпада..."
XORG_CONF_DIR="/etc/X11/xorg.conf.d"
sudo mkdir -p "$XORG_CONF_DIR"

XORG_CONF="$XORG_CONF_DIR/40-libinput.conf"
sudo tee "$XORG_CONF" > /dev/null << 'XORG'
Section "InputClass"
    Identifier "libinput touchpad"
    Driver "libinput"
    MatchIsTouchpad "on"
    Option "Tapping" "on"
    Option "ClickMethod" "clickfinger"
    Option "NaturalScrolling" "true"
    Option "DisableWhileTyping" "true"
EndSection

Section "InputClass"
    Identifier "libinput touchscreen"
    Driver "libinput"
    MatchIsTouchscreen "on"
    Option "Tapping" "on"
EndSection
XORG
success "Xorg конфиг: $XORG_CONF"

# ============================================================
# Шаг 6. Создание директорий конфигов
# ============================================================
info "Создание директорий конфигов..."
mkdir -p \
    ~/.config/bspwm \
    ~/.config/sxhkd \
    ~/.config/polybar \
    ~/.config/rofi
success "Директории готовы"

# ============================================================
# Шаг 7. bspwmrc
# ============================================================
info "Запись bspwmrc..."
cat > ~/.config/bspwm/bspwmrc << 'BSPWMRC'
#!/bin/bash

# === Внешний вид ===
bspc config border_width         2
bspc config window_gap           8
bspc config split_ratio          0.52
bspc config focus_follows_pointer true
bspc config pointer_modifier     mod4
bspc config automatic_scheme     spiral
bspc config borderless_monocle   true
bspc config gapless_monocle      true

# === Рабочие столы ===
bspc monitor -d I II III IV V VI VII VIII IX X

# === Правила окон ===
bspc rule -a Gimp desktop='^8' state=floating follow=on
bspc rule -a Chromium desktop='^2'

# === Курсор ===
xsetroot -cursor_name left_ptr

# === Автозапуск ===
sxhkd &
picom -b &
~/.config/polybar/launch.sh &
BSPWMRC

# Добавить демон жестов
if [[ -n "$GESTURE_DAEMON" ]]; then
    echo "$GESTURE_DAEMON &" >> ~/.config/bspwm/bspwmrc
fi

# Добавить экранную клавиатуру
if [[ "$INSTALL_KEYBOARD" == "yes" ]]; then
    echo "onboard &" >> ~/.config/bspwm/bspwmrc
fi

chmod +x ~/.config/bspwm/bspwmrc
success "bspwmrc готов"

# ============================================================
# Шаг 8. sxhkdrc
# ============================================================
info "Запись sxhkdrc..."
cat > ~/.config/sxhkd/sxhkdrc << SXHKDRC
# Перезагрузка конфигов
super + Escape
    pkill -USR1 -x sxhkd

# Перезапуск / выход из bspwm
super + shift + {r,q}
    bspc {wm -r,quit}

# Терминал
super + Return
    $TERMINAL

# Запуск rofi
super + d
    rofi -show drun -theme ~/.config/rofi/touch.rasi

# Закрыть окно
super + {_,shift +}c
    bspc node -{c,k}

# Режим окна
super + {t,shift + t,f}
    bspc node -t {tiled,pseudo_tiled,floating}

super + space
    bspc node -t fullscreen

# Фокус (hjkl)
super + {h,j,k,l}
    bspc node -f {west,south,north,east}

# Фокус (стрелки)
super + {Left,Down,Up,Right}
    bspc node -f {west,south,north,east}

# Перемещение окон
super + shift + {h,j,k,l}
    bspc node -s {west,south,north,east}

# Рабочие столы
super + {1-9,0}
    bspc desktop -f '^{1-9,10}'

super + shift + {1-9,0}
    bspc node -d '^{1-9,10}'

# Монокль
super + m
    bspc desktop -l next

# Изменение размера
super + alt + {h,j,k,l}
    bspc node -z {left -20 0,bottom 0 20,top 0 -20,right 20 0}

# Плавающие окна — перемещение
super + {Left,Down,Up,Right}
    bspc node -v {-20 0,0 20,0 -20,20 0}
SXHKDRC
success "sxhkdrc готов"

# ============================================================
# Шаг 9. polybar launch script
# ============================================================
info "Запись polybar launch.sh..."
cat > ~/.config/polybar/launch.sh << 'LAUNCH'
#!/usr/bin/env bash
killall -q polybar 2>/dev/null
while pgrep -x polybar >/dev/null; do sleep 0.1; done
polybar -r example >> /tmp/polybar-example.log 2>&1 &
LAUNCH
chmod +x ~/.config/polybar/launch.sh
success "polybar launch.sh готов"

# ============================================================
# Шаг 10. polybar config
# ============================================================
info "Запись polybar config.ini..."

KEYBOARD_MODULE=""
if [[ "$INSTALL_KEYBOARD" == "yes" ]]; then
    KEYBOARD_MODULE="touch-keyboard"
fi

cat > ~/.config/polybar/config.ini << POLYCONF
[colors]
background = #1e1e2e
foreground = #cdd6f4
primary = #89b4fa
secondary = #f5c2e7
alert = #f38ba8
disabled = #6c7086

[bar/example]
width = 100%
height = 32pt
radius = 0
background = \${colors.background}
foreground = \${colors.foreground}
border-size = 0
padding-left = 4
padding-right = 4
module-margin-left = 1
module-margin-right = 1

font-0 = "JetBrainsMono Nerd Font:size=10;2"

modules-left = bspwm
modules-center = xwindow
modules-right = touch-launcher pulseaudio memory cpu date $KEYBOARD_MODULE touch-power

cursor-click = pointer
enable-ipc = true

[module/bspwm]
type = internal/bspwm
label-focused = %name%
label-focused-background = \${colors.primary}
label-focused-foreground = \${colors.background}
label-focused-padding = 2
label-occupied = %name%
label-occupied-padding = 2
label-empty = %name%
label-empty-foreground = \${colors.disabled}
label-empty-padding = 2
label-urgent = %name%
label-urgent-background = \${colors.alert}
label-urgent-padding = 2

[module/xwindow]
type = internal/xwindow
label = %title%
label-maxlen = 50

[module/pulseaudio]
type = internal/pulseaudio
format-volume = <label-volume> <ramp-volume>
label-volume = %percentage%%
label-muted = " MUTE"
ramp-volume-0 = ▁
ramp-volume-1 = ▂
ramp-volume-2 = ▃
ramp-volume-3 = ▄
ramp-volume-4 = ▅
ramp-volume-5 = ▆
ramp-volume-6 = ▇
ramp-volume-7 = █
click-right = pavucontrol

[module/memory]
type = internal/memory
label = RAM %percentage_used%%
format-prefix = " "
format-prefix-foreground = \${colors.primary}

[module/cpu]
type = internal/cpu
label = CPU %percentage%%
format-prefix = " "
format-prefix-foreground = \${colors.secondary}

[module/date]
type = internal/date
interval = 1
date = %H:%M
date-alt = %Y-%m-%d %H:%M
label = %date%
format-prefix = " "
format-prefix-foreground = \${colors.primary}

[module/touch-launcher]
type = custom/text
content = " Apps "
content-background = \${colors.primary}
content-foreground = \${colors.background}
content-padding = 2
click-left = rofi -show drun -theme ~/.config/rofi/touch.rasi

[module/touch-power]
type = custom/text
content = " Power "
content-background = \${colors.alert}
content-foreground = \${colors.background}
content-padding = 2
click-left = rofi -show power -modi power:~/.local/bin/rofi-power-menu -theme ~/.config/rofi/touch.rasi
POLYCONF

if [[ "$INSTALL_KEYBOARD" == "yes" ]]; then
    cat >> ~/.config/polybar/config.ini << 'KBMOD'

[module/touch-keyboard]
type = custom/text
content = " ABC "
content-background = ${colors.secondary}
content-foreground = ${colors.background}
content-padding = 2
click-left = onboard --show
KBMOD
fi

success "polybar config.ini готов"

# ============================================================
# Шаг 11. rofi touch theme
# ============================================================
info "Запись rofi touch.rasi..."
cat > ~/.config/rofi/touch.rasi << 'ROFI'
* {
    background: #1e1e2e;
    foreground: #cdd6f4;
    selected: #89b4fa;
    active: #a6e3a1;
    urgent: #f38ba8;
    font: "JetBrainsMono Nerd Font 16";
    element-padding: 12;
    element-border: 0;
}

window {
    width: 40%;
    padding: 20;
    background-color: @background;
    border-radius: 10;
}

entry {
    placeholder: "Поиск...";
    padding: 12px;
    font: @font;
    text-color: @foreground;
}

listview {
    lines: 8;
    columns: 1;
    padding: 8px 0;
}

element {
    padding: 16px 12px;
    border-radius: 6;
    font: @font;
}

element selected {
    background-color: @selected;
    text-color: @background;
}

element-text {
    vertical-align: 0.5;
    horizontal-align: 0.0;
}
ROFI

# Базовый config.rasi
cat > ~/.config/rofi/config.rasi << 'ROFI_BASE'
configuration {
    modi: "drun,run,window";
    font: "JetBrainsMono Nerd Font 12";
    show-icons: true;
}
ROFI_BASE
success "rofi темы готовы"

# ============================================================
# Шаг 12. Меню питания для rofi
# ============================================================
info "Создание скрипта меню питания..."
mkdir -p ~/.local/bin
cat > ~/.local/bin/rofi-power-menu << 'POWERMENU'
#!/bin/bash
# rofi-power-menu — простое меню питания для rofi
# Использование: rofi -show power -modi power:rofi-power-menu

entries="\
Logout
Reboot
Shutdown
Suspend
Hibernate"

selected=$(echo -e "$entries" | rofi -dmenu -i -p "Power:" -theme ~/.config/rofi/touch.rasi)

case "$selected" in
    Logout)    bspc quit ;;
    Reboot)    systemctl reboot ;;
    Shutdown)  systemctl poweroff ;;
    Suspend)   systemctl suspend ;;
    Hibernate) systemctl hibernate ;;
esac
POWERMENU
chmod +x ~/.local/bin/rofi-power-menu
success "Меню питания готово"

# ============================================================
# Шаг 13. Конфиг жестов
# ============================================================
if [[ "$GESTURE_TOOL" == "libinput-gestures" ]]; then
    info "Запись libinput-gestures.conf..."
    cat > ~/.config/libinput-gestures.conf << 'GESTURES'
# 3 пальца — навигация
gesture swipe up 3
    bspc desktop -f next

gesture swipe down 3
    bspc desktop -f prev

gesture swipe left 3
    bspc node -f west

gesture swipe right 3
    bspc node -f east

# 4 пальца — управление окнами
gesture swipe up 4
    bspc desktop -l next

gesture swipe down 4
    bspc node -t tiled

gesture swipe left 4
    bspc node -d prev

gesture swipe right 4
    bspc node -d next

# Pinch — действия
gesture pinch in 2
    bspc node -c

gesture pinch out 2
    rofi -show drun -theme ~/.config/rofi/touch.rasi
GESTURES
    success "libinput-gestures.conf готов"

    # Автозапуск
    libinput-gestures-setup autostart 2>/dev/null || true

elif [[ "$GESTURE_TOOL" == "fusuma" ]]; then
    info "Запись fusuma config.yml..."
    mkdir -p ~/.config/fusuma
    cat > ~/.config/fusuma/config.yml << 'FUSUMA'
swipe:
  3:
    left:
      command: 'bspc node -f west'
    right:
      command: 'bspc node -f east'
    up:
      command: 'bspc desktop -f next'
    down:
      command: 'bspc desktop -f prev'
  4:
    left:
      command: 'bspc node -d prev'
    right:
      command: 'bspc node -d next'
    up:
      command: 'bspc desktop -l next'
    down:
      command: 'bspc node -t tiled'

pinch:
  2:
    in:
      command: 'bspc node -c'
    out:
      command: 'rofi -show drun -theme ~/.config/rofi/touch.rasi'

hold:
  3:
    command: 'xdotool key super+d'

threshold:
  swipe: 0.4
  pinch: 0.3

interval:
  swipe: 0.8
  pinch: 0.1
FUSUMA
    success "fusuma config.yml готов"
fi

# ============================================================
# Шаг 14. xinitrc
# ============================================================
info "Запись .xinitrc..."
cat > ~/.xinitrc << 'XINIT'
#!/bin/sh
setxkbmap -layout us,ru -variant , -option grp:alt_shift_toggle &
exec bspwm
XINIT
chmod +x ~/.xinitrc
success ".xinitrc готов"

# ============================================================
# Шаг 15. Автологин (опционально)
# ============================================================
echo ""
warn "Настроить автологин на tty1 + автозапуск X?"
warn "Это автоматически запустит графическую сессию при включении."
read -rp "Автологин? [y/N]: " autologin

if [[ "${autologin,,}" == "y" ]]; then
    info "Настройка автологина..."
    OVERRIDE_DIR="/etc/systemd/system/getty@tty1.service.d"
    sudo mkdir -p "$OVERRIDE_DIR"
    sudo tee "$OVERRIDE_DIR/autologin.conf" > /dev/null << AUTOLOGIN
[Service]
ExecStart=
ExecStart=-/usr/bin/agetty --autologin $USER --noclear %I \$TERM
AUTOLOGIN

    # Добавить в .bash_profile
    BASH_PROFILE="$HOME/.bash_profile"
    TOUCH_FILE="$HOME/.config/.bspwm-autostart-marker"

    if ! grep -q "startx" "$BASH_PROFILE" 2>/dev/null; then
        cat >> "$BASH_PROFILE" << 'AUTOSTARTX'
if [ -z "$DISPLAY" ] && [ "$XDG_VTNR" = 1 ]; then
    exec startx
fi
AUTOSTARTX
    fi
    success "Автологин настроен"
else
    info "Автологин пропущен. Запускать графическую сессию: startx"
fi

# ============================================================
# Шаг 16. Финал
# ============================================================
echo ""
echo -e "${GREEN}=========================================="
echo " Установка завершена!"
echo "==========================================${NC}"
echo ""
echo " Установлено:"
echo "   • Xorg + libinput (тачскрин/тачпад)"
echo "   • bspwm + sxhkd"
echo "   • polybar (с тач-кнопками)"
echo "   • rofi (тач-тема)"
echo "   • $TERMINAL (терминал)"
echo "   • picom (композитор)"
if [[ "$INSTALL_KEYBOARD" == "yes" ]]; then
echo "   • onboard (экранная клавиатура)"
fi
if [[ "$GESTURE_TOOL" == "libinput-gestures" ]]; then
echo "   • libinput-gestures (жесты)"
elif [[ "$GESTURE_TOOL" == "fusuma" ]]; then
echo "   • fusuma (жесты)"
fi
echo ""
echo " Конфиги:"
echo "   ~/.config/bspwm/bspwmrc"
echo "   ~/.config/sxhkd/sxhkdrc"
echo "   ~/.config/polybar/config.ini"
echo "   ~/.config/polybar/launch.sh"
echo "   ~/.config/rofi/touch.rasi"
echo "   ~/.config/rofi/config.rasi"
if [[ "$GESTURE_TOOL" == "libinput-gestures" ]]; then
echo "   ~/.config/libinput-gestures.conf"
elif [[ "$GESTURE_TOOL" == "fusuma" ]]; then
echo "   ~/.config/fusuma/config.yml"
fi
echo "   ~/.local/bin/rofi-power-menu"
echo "   /etc/X11/xorg.conf.d/40-libinput.conf"
echo ""
echo " Горячие клавиши:"
echo "   Super+Enter  — терминал"
echo "   Super+d      — rofi (лаунчер)"
echo "   Super+1..0   — переключение рабочих столов"
echo "   Super+hjkl   — фокус между окнами"
echo "   Super+Shift+c— закрыть окно"
echo "   Super+Space  — полноэкранный режим"
echo "   Super+m      — монокль"
echo "   Super+Esc    — перезагрузить sxhkd"
echo "   Super+Shift+r— перезапустить bspwm"
echo "   Super+Shift+q— выход из bspwm"
echo ""

if [[ "$GESTURE_TOOL" != "none" ]]; then
echo " Тач-жесты:"
echo "   3 пальца ↑↓  — переключение рабочих столов"
echo "   3 пальца ←→  — фокус между окнами"
echo "   4 пальца ↑↓  — монокль / tiled"
echo "   4 пальца ←→  — перенос окна на соседний стол"
echo "   Pinch в      — закрыть окно"
echo "   Pinch из     — запустить rofi"
if [[ "$GESTURE_TOOL" == "fusuma" ]]; then
echo "   Hold 3 пальца — запустить rofi"
fi
echo ""
fi

warn "Перезагрузись для применения прав группы input и udev правил."
warn "После перезагрузки запусти: startx"
echo ""
read -rp "Перезагрузить сейчас? [y/N]: " reboot_now
if [[ "${reboot_now,,}" == "y" ]]; then
    sudo reboot
fi
