#!/usr/bin/env bash
#
# install-bspwm-touch.sh
# Полная установка bspwm + polybar + rofi + тач-управление на Arch Linux
#
# Использование:
#   ./install-bspwm-touch.sh                          интерактивный режим
#   ./install-bspwm-touch.sh --gestures fusuma        жесты через fusuma
#   ./install-bspwm-touch.sh --gestures libinput       жесты через libinput-gestures
#   ./install-bspwm-touch.sh --gestures none          без жестов
#   ./install-bspwm-touch.sh --terminal kitty         выбор терминала
#   ./install-bspwm-touch.sh --auto-login             включить автологин
#   ./install-bspwm-touch.sh --non-interactive         без вопросов
#   ./install-bspwm-touch.sh --build-iso              собрать кастомный ISO
#   ./install-bspwm-touch.sh --help                   справка
#

set -euo pipefail

# ─── Цвета ───────────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

info()    { echo -e "${CYAN}[INFO]${NC}  $*"; }
success() { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error()   { echo -e "${RED}[ERR]${NC}   $*"; exit 1; }

# ─── Переменные по умолчанию ─────────────────────────────────────────────────
GESTURES="libinput"
TERMINAL="alacritty"
AUTO_LOGIN=false
NON_INTERACTIVE=false
BUILD_ISO=false

# ─── Парсинг аргументов ──────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --gestures)
            GESTURES="$2"; shift 2 ;;
        --gestures=*)
            GESTURES="${1#*=}"; shift ;;
        --terminal)
            TERMINAL="$2"; shift 2 ;;
        --terminal=*)
            TERMINAL="${1#*=}"; shift ;;
        --auto-login)
            AUTO_LOGIN=true; shift ;;
        --non-interactive)
            NON_INTERACTIVE=true; shift ;;
        --build-iso)
            BUILD_ISO=true; shift ;;
        --help|-h)
            cat << 'HELP'
install-bspwm-touch.sh — установка bspwm + polybar + rofi + тач-управление

Использование:
  install-bspwm-touch.sh [ОПЦИИ]

Опции:
  --gestures <libinput|fusuma|none>   Инструмент жестов (по умолчанию: libinput)
  --terminal <name>                   Терминал: alacritty, kitty, st (по умолчанию: alacritty)
  --auto-login                        Включить автологин + автозапуск X на tty1
  --non-interactive                   Не задавать вопросов
  --build-iso                         Собрать кастомный Arch ISO с предустановкой
  --help, -h                          Эта справка

Примеры:
  install-bspwm-touch.sh --gestures fusuma --terminal kitty --auto-login --non-interactive
  install-bspwm-touch.sh --build-iso
HELP
            exit 0 ;;
        *)
            error "Неизвестная опция: $1 (используйте --help)" ;;
    esac
done

# ─── Проверки ─────────────────────────────────────────────────────────────────
check_os() {
    if [[ ! -f /etc/arch-release ]]; then
        error "Этот скрипт предназначен для Arch Linux. /etc/arch-release не найден."
    fi
    success "Arch Linux подтверждён"
}

check_not_root() {
    if [[ $EUID -eq 0 ]]; then
        error "Не запускайте от root. Используйте sudo внутри скрипта."
    fi
    success "Запущено от пользователя: $(whoami)"
}

check_sudo() {
    if ! sudo -v >/dev/null 2>&1; then
        error "Нужны права sudo. Добавьте пользователя в wheel: 'sudo usermod -aG wheel $USER'"
    fi
    success "sudo доступен"
}

# ─── Интерактивные вопросы ────────────────────────────────────────────────────
ask_questions() {
    if [[ "$NON_INTERACTIVE" == true ]]; then
        info "Неинтерактивный режим: gestures=$GESTURES terminal=$TERMINAL auto_login=$AUTO_LOGIN"
        return
    fi

    echo -e "\n${BOLD}Настройка установки${NC}\n"

    # Жесты
    read -rp "Инструмент жестов [libinput/fusuma/none] (по умолчанию: $GESTURES): " ans
    case "${ans,,}" in
        fusuma)     GESTURES="fusuma" ;;
        libinput)   GESTURES="libinput" ;;
        none)       GESTURES="none" ;;
        "")         ;;
        *)          warn "Не распознано, используется $GESTURES" ;;
    esac

    # Терминал
    read -rp "Терминал [alacritty/kitty/st] (по умолчанию: $TERMINAL): " ans
    case "${ans,,}" in
        alacritty|kitty|st) TERMINAL="${ans,,}" ;;
        "")                 ;;
        *)                  warn "Не распознано, используется $TERMINAL" ;;
    esac

    # Автологин
    read -rp "Включить автологин + автозапуск X? [y/N]: " ans
    case "${ans,,}" in
        y|yes) AUTO_LOGIN=true ;;
    esac

    echo
    info "Параметры установки:"
    info "  Жесты:     $GESTURES"
    info "  Терминал:  $TERMINAL"
    info "  Автологин: $AUTO_LOGIN"
    echo

    read -rp "Продолжить? [Y/n]: " ans
    case "${ans,,}" in
        n|no) error "Установка отменена пользователем" ;;
    esac
}

# ─── Установка пакетов из pacman ─────────────────────────────────────────────
install_pacman_packages() {
    info "Обновление системы..."
    sudo pacman -Syu --noconfirm

    info "Установка базовых пакетов..."
    local pkgs=(
        xorg-server xorg-xinit xorg-apps
        xf86-input-libinput xdotool wmctrl
        bspwm sxhkd polybar rofi
        picom
        ttf-jetbrains-mono-nerd noto-fonts-emoji
        onboard
        git base-devel
    )

    # Терминал
    case "$TERMINAL" in
        alacritty) pkgs+=("alacritty") ;;
        kitty)     pkgs+=("kitty") ;;
        st)        pkgs+=("st") ;;
    esac

    sudo pacman -S --noconfirm --needed "${pkgs[@]}"
    success "Пакеты из pacman установлены"
}

# ─── Установка AUR-хелпера ────────────────────────────────────────────────────
install_yay() {
    if command -v yay &>/dev/null; then
        success "yay уже установлен"
        return
    fi

    info "Установка yay из AUR..."
    local tmpdir
    tmpdir=$(mktemp -d)
    git clone https://aur.archlinux.org/yay.git "$tmpdir/yay"
    (cd "$tmpdir/yay" && makepkg -si --noconfirm)
    rm -rf "$tmpdir"
    success "yay установлен"
}

# ─── Установка инструментов жестов ────────────────────────────────────────────
install_gestures() {
    case "$GESTURES" in
        libinput)
            info "Установка libinput-gestures из AUR..."
            yay -S --noconfirm --needed libinput-gestures
            success "libinput-gestures установлен"
            ;;
        fusuma)
            info "Установка fusuma..."
            sudo pacman -S --noconfirm --needed ruby libinput
            sudo gem install fusuma
            success "fusuma установлен"
            ;;
        none)
            warn "Установка жестов пропущена"
            ;;
    esac
}

# ─── Настройка прав тача ──────────────────────────────────────────────────────
setup_touch_permissions() {
    info "Настройка прав тачскрина/тачпада..."

    sudo gpasswd -a "$USER" input 2>/dev/null || true

    sudo tee /etc/udev/rules.d/71-touchpad.rules > /dev/null << 'UDEV'
ACTION!="remove", ENV{ID_INPUT_TOUCHPAD}=="1", TAG+="uaccess"
ACTION!="remove", ENV{ID_INPUT_TOUCHSCREEN}=="1", TAG+="uaccess"
UDEV

    success "Права тача настроены"
}

# ─── Xorg-конфиг для тача ──────────────────────────────────────────────────────
setup_xorg_touch() {
    info "Создание Xorg-конфига для тача..."

    sudo mkdir -p /etc/X11/xorg.conf.d
    sudo tee /etc/X11/xorg.conf.d/40-libinput.conf > /dev/null << 'XORG'
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

    success "Xorg-конфиг для тача создан"
}

# ─── Создание директорий ──────────────────────────────────────────────────────
create_dirs() {
    info "Создание директорий конфигов..."
    mkdir -p \
        ~/.config/bspwm \
        ~/.config/sxhkd \
        ~/.config/polybar \
        ~/.config/rofi \
        ~/.local/bin
    success "Директории готовы"
}

# ─── bspwmrc ──────────────────────────────────────────────────────────────────
write_bspwmrc() {
    info "Запись bspwmrc..."

    # Определяем команду запуска жестов
    local gestures_cmd=""
    case "$GESTURES" in
        libinput) gestures_cmd="libinput-gestures-setup start" ;;
        fusuma)   gestures_cmd="fusuma -d" ;;
    esac

    cat > ~/.config/bspwm/bspwmrc << BSPWMRC
#!/bin/bash

# ─── Внешний вид ──────────────────────────────────────────────────
bspc config border_width         2
bspc config window_gap           8
bspc config split_ratio          0.52
bspc config focus_follows_pointer true
bspc config pointer_modifier     mod4
bspc config automatic_scheme     spiral
bspc config borderless_monocle   true
bspc config gapless_monocle      true

# ─── Цвета ────────────────────────────────────────────────────────
bspc config normal_border_color   "#1e1e2e"
bspc config active_border_color   "#89b4fa"
bspc config focused_border_color  "#89b4fa"
bspc config presel_feedback_color "#f5c2e7"

# ─── Рабочие столы ─────────────────────────────────────────────────
bspc monitor -d I II III IV V VI VII VIII IX X

# ─── Правила окон ──────────────────────────────────────────────────
bspc rule -a Gimp desktop='^8' state=floating follow=on
bspc rule -a Chromium desktop='^2'
bspc rule -a onboard state=floating sticky=on

# ─── Курсор ──────────────────────────────────────────────────────
xsetroot -cursor_name left_ptr

# ─── Автозапуск ───────────────────────────────────────────────────
sxhkd &
picom -b &
~/.config/polybar/launch.sh &
${gestures_cmd}
BSPWMRC

    chmod +x ~/.config/bspwm/bspwmrc
    success "bspwmrc записан"
}

# ─── sxhkdrc ──────────────────────────────────────────────────────────────────
write_sxhkdrc() {
    info "Запись sxhkdrc..."

    cat > ~/.config/sxhkd/sxhkdrc << 'SXHKD'
# ─── Перезагрузка и выход ──────────────────────────────────────────
super + Escape
    pkill -USR1 -x sxhkd

super + shift + {r,q}
    bspc {wm -r,quit}

# ─── Терминал ──────────────────────────────────────────────────────
super + Return
    __TERMINAL__

# ─── Rofi (лаунчер) ────────────────────────────────────────────────
super + d
    rofi -show drun -theme ~/.config/rofi/touch.rasi

super + shift + d
    rofi -show run -theme ~/.config/rofi/touch.rasi

# ─── Закрыть окно ──────────────────────────────────────────────────
super + {_,shift +}c
    bspc node -{c,k}

# ─── Режим окна ────────────────────────────────────────────────────
super + {t,shift + t,f,space}
    bspc node -t {tiled,pseudo_tiled,floating,fullscreen}

# ─── Фокус ────────────────────────────────────────────────────────
super + {h,j,k,l}
    bspc node -f {west,south,north,east}

super + {Left,Down,Up,Right}
    bspc node -f {west,south,north,east}

# ─── Перемещение окон ──────────────────────────────────────────────
super + shift + {h,j,k,l}
    bspc node -s {west,south,north,east}

# ─── Рабочие столы ──────────────────────────────────────────────────
super + {1-9,0}
    bspc desktop -f '^{1-9,10}'

super + shift + {1-9,0}
    bspc node -d '^{1-9,10}'

# ─── Монокль ──────────────────────────────────────────────────────
super + m
    bspc desktop -l next

# ─── Размер окна ───────────────────────────────────────────────────
super + alt + {h,j,k,l}
    bspc node -z {left -20 0,bottom 0 20,top 0 -20,right 20 0}

# ─── Плавающие окна ────────────────────────────────────────────────
super + {_,shift + }{Left,Down,Up,Right}
    bspc node -{f,v} {west,south,north,east}

# ─── Экранная клавиатура ───────────────────────────────────────────
super + shift + k
    onboard
SXHKD

    # Подставляем терминал
    sed -i "s|__TERMINAL__|$TERMINAL|" ~/.config/sxhkd/sxhkdrc
    success "sxhkdrc записан"
}

# ─── polybar ──────────────────────────────────────────────────────────────────
write_polybar() {
    info "Запись polybar конфигов..."

    # launch.sh
    cat > ~/.config/polybar/launch.sh << 'LAUNCH'
#!/usr/bin/env bash
killall -q polybar
while pgrep -u $UID -x polybar > /dev/null; do sleep 0.5; done
polybar -r example >> /tmp/polybar-example.log 2>&1 &
LAUNCH
    chmod +x ~/.config/polybar/launch.sh

    # config.ini
    cat > ~/.config/polybar/config.ini << 'POLYBAR'
[colors]
background = #1e1e2e
foreground = #cdd6f4
primary    = #89b4fa
secondary  = #f5c2e7
alert      = #f38ba8
disabled   = #6c7086

[bar/example]
width = 100%
height = 36pt
radius = 0
background = ${colors.background}
foreground = ${colors.foreground}
border-size = 0
padding-left = 4
padding-right = 4
module-margin-left = 1
module-margin-right = 1

font-0 = "JetBrainsMono Nerd Font:size=10;2"

modules-left = bspwm
modules-center = xwindow
modules-right = touch-launcher pulseaudio memory cpu date touch-power

cursor-click = pointer
enable-ipc = true

[module/bspwm]
type = internal/bspwm
label-focused = %name%
label-focused-background = ${colors.primary}
label-focused-foreground = ${colors.background}
label-focused-padding = 3
label-occupied = %name%
label-occupied-padding = 3
label-empty = %name%
label-empty-foreground = ${colors.disabled}
label-empty-padding = 3
label-urgent = %name%
label-urgent-background = ${colors.alert}
label-urgent-padding = 3

[module/xwindow]
type = internal/xwindow
label = %title%
label-maxlen = 50

[module/touch-launcher]
type = custom/text
content = " Apps "
content-background = ${colors.primary}
content-foreground = ${colors.background}
content-padding = 2
click-left = rofi -show drun -theme ~/.config/rofi/touch.rasi

[module/touch-power]
type = custom/text
content = " Power "
content-background = ${colors.alert}
content-foreground = ${colors.background}
content-padding = 2
click-left = rofi -show p -modi p:~/.local/bin/rofi-power-menu -theme ~/.config/rofi/touch.rasi

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

[module/memory]
type = internal/memory
label = RAM %percentage_used%%
format-prefix = " "
format-prefix-foreground = ${colors.primary}

[module/cpu]
type = internal/cpu
label = CPU %percentage%%
format-prefix = " "
format-prefix-foreground = ${colors.secondary}

[module/date]
type = internal/date
interval = 1
date = %H:%M
date-alt = %Y-%m-%d %H:%M
label = %date%
format-prefix = " "
format-prefix-foreground = ${colors.primary}
POLYBAR

    success "polybar настроен"
}

# ─── rofi ──────────────────────────────────────────────────────────────────────
write_rofi() {
    info "Запись rofi темы..."

    cat > ~/.config/rofi/touch.rasi << 'ROFI'
* {
    background: #1e1e2e;
    foreground: #cdd6f4;
    selected:   #89b4fa;
    active:     #a6e3a1;
    urgent:     #f38ba8;
    font:       "JetBrainsMono Nerd Font 16";
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

    success "rofi тема создана"
}

# ─── Меню питания ─────────────────────────────────────────────────────────────
write_power_menu() {
    info "Создание меню питания..."

    cat > ~/.local/bin/rofi-power-menu << 'POWER'
#!/usr/bin/env bash
# Меню питания для rofi

entries="Lock\nLogout\nReboot\nShutdown\nSuspend\nHibernate"

selected=$(echo -e "$entries" | rofi -dmenu -theme ~/.config/rofi/touch.rasi -p "Power")

case "$selected" in
    Lock)     loginctl lock-session ;;
    Logout)   bspc quit ;;
    Reboot)   systemctl reboot ;;
    Shutdown) systemctl poweroff ;;
    Suspend)  systemctl suspend ;;
    Hibernate) systemctl hibernate ;;
esac
POWER

    chmod +x ~/.local/bin/rofi-power-menu
    success "Меню питания готово"
}

# ─── Конфиг жестов ────────────────────────────────────────────────────────────
write_gestures_config() {
    case "$GESTURES" in
        libinput)
            info "Запись libinput-gestures.conf..."
            cat > ~/.config/libinput-gestures.conf << 'GESTURES'
# 3 пальца — навигация между окнами и столами
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

# Pinch — закрыть / запустить rofi
gesture pinch in 2
    bspc node -c

gesture pinch out 2
    rofi -show drun -theme ~/.config/rofi/touch.rasi
GESTURES
            success "libinput-gestures.conf записан"
            ;;

        fusuma)
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
            success "fusuma config.yml записан"
            ;;
        none)
            info "Конфиг жестов пропущен"
            ;;
    esac
}

# ─── xinitrc ──────────────────────────────────────────────────────────────────
write_xinitrc() {
    info "Запись .xinitrc..."

    cat > ~/.xinitrc << 'XINIT'
#!/bin/sh
setxkbmap -layout us,ru -variant , -option grp:alt_shift_toggle &
exec bspwm
XINIT

    chmod +x ~/.xinitrc
    success ".xinitrc записан"
}

# ─── Автологин ─────────────────────────────────────────────────────────────────
setup_autologin() {
    if [[ "$AUTO_LOGIN" != true ]]; then
        return
    fi

    info "Настройка автологина..."

    local user
    user=$(whoami)

    sudo mkdir -p /etc/systemd/system/getty@tty1.service.d
    sudo tee /etc/systemd/system/getty@tty1.service.d/autologin.conf > /dev/null << AUTOLOGIN
[Service]
ExecStart=
ExecStart=-/usr/bin/agetty --autologin $user --noclear %I \$TERM
AUTOLOGIN

    # Автозапуск X из bash_profile
    if ! grep -q 'startx' ~/.bash_profile 2>/dev/null; then
        cat >> ~/.bash_profile << 'BASH'

if [ -z "$DISPLAY" ] && [ "$XDG_VTNR" = 1 ]; then
    exec startx
fi
BASH
    fi

    success "Автологин настроен на tty1"
}

# ─── Сборка ISO ───────────────────────────────────────────────────────────────
build_iso() {
    info "Сборка кастомного Arch ISO..."

    local iso_dir="$HOME/arch-bspwm-iso"
    sudo pacman -S --noconfirm --needed archiso

    # Базовый профиль
    sudo rm -rf "$iso_dir"
    cp -r /usr/share/archiso/configs/releng "$iso_dir"

    # Пакеты для ISO
    local packages=(
        xorg-server xorg-xinit xorg-apps
        xf86-input-libinput xdotool wmctrl
        bspwm sxhkd polybar rofi
        picom alacritty
        ttf-jetbrains-mono-nerd noto-fonts-emoji
        onboard
        ruby
    )

    for pkg in "${packages[@]}"; do
        echo "$pkg" | sudo tee -a "$iso_dir/packages.x86_64" > /dev/null
    done

    # Конфиги в /etc/skel
    local skel="$iso_dir/airootfs/etc/skel"
    sudo mkdir -p \
        "$skel/.config/bspwm" \
        "$skel/.config/sxhkd" \
        "$skel/.config/polybar" \
        "$skel/.config/rofi" \
        "$skel/.local/bin"

    # Копируем конфиги из домашней директории
    sudo cp ~/.config/bspwm/bspwmrc        "$skel/.config/bspwm/bspwmrc"
    sudo cp ~/.config/sxhkd/sxhkdrc        "$skel/.config/sxhkd/sxhkdrc"
    sudo cp ~/.config/polybar/config.ini   "$skel/.config/polybar/config.ini"
    sudo cp ~/.config/polybar/launch.sh    "$skel/.config/polybar/launch.sh"
    sudo cp ~/.config/rofi/touch.rasi      "$skel/.config/rofi/touch.rasi"
    sudo cp ~/.local/bin/rofi-power-menu   "$skel/.local/bin/rofi-power-menu"
    sudo cp ~/.xinitrc                      "$skel/.xinitrc"

    sudo chmod +x "$skel/.config/bspwm/bspwmrc"
    sudo chmod +x "$skel/.config/polybar/launch.sh"
    sudo chmod +x "$skel/.local/bin/rofi-power-menu"
    sudo chmod +x "$skel/.xinitrc"

    # Конфиг жестов
    if [[ "$GESTURES" == "libinput" ]]; then
        sudo mkdir -p "$skel/.config"
        sudo cp ~/.config/libinput-gestures.conf "$skel/.config/libinput-gestures.conf"
        echo "libinput-gestures" | sudo tee -a "$iso_dir/packages.x86_64" > /dev/null
    elif [[ "$GESTURES" == "fusuma" ]]; then
        sudo mkdir -p "$skel/.config/fusuma"
        sudo cp ~/.config/fusuma/config.yml "$skel/.config/fusuma/config.yml"
    fi

    # Xorg-конфиг
    sudo mkdir -p "$iso_dir/airootfs/etc/X11/xorg.conf.d"
    sudo cp /etc/X11/xorg.conf.d/40-libinput.conf \
           "$iso_dir/airootfs/etc/X11/xorg.conf.d/40-libinput.conf"

    # udev-правила
    sudo mkdir -p "$iso_dir/airootfs/etc/udev/rules.d"
    sudo cp /etc/udev/rules.d/71-touchpad.rules \
           "$iso_dir/airootfs/etc/udev/rules.d/71-touchpad.rules"

    # Автологин для live-сессии
    sudo tee "$iso_dir/airootfs/etc/systemd/system/getty@tty1.service.d/autologin.conf" > /dev/null << 'LIVE'
[Service]
ExecStart=
ExecStart=-/usr/bin/agetty --autologin root --noclear %I $TERM
LIVE

    # Автозапуск X
    sudo tee -a "$iso_dir/airootfs/root/.bash_profile" > /dev/null << 'LIVEX'

if [ -z "$DISPLAY" ] && [ "$XDG_VTNR" = 1 ]; then
    exec startx
fi
LIVEX

    # Установка fusuma в ISO
    if [[ "$GESTURES" == "fusuma" ]]; then
        sudo tee -a "$iso_dir/airootfs/root/.zshrc" > /dev/null << 'GEM'

# Установка fusuma при первом запуске
if ! command -v fusuma &>/dev/null; then
    gem install fusuma 2>/dev/null
fi
GEM
    fi

    # Сборка
    info "Запуск mkarchiso (может занять 10-20 минут)..."
    cd "$iso_dir"
    sudo mkarchiso -v -w /tmp/archiso-work -o out .
    success "ISO собран: $iso_dir/out/"

    # Права
    sudo chown -R "$USER:$USER" "$iso_dir/out/"

    echo
    info "Запись на флешку:"
    echo "  sudo dd if=$iso_dir/out/*.iso of=/dev/sdX bs=4M status=progress"
}

# ─── Финал ────────────────────────────────────────────────────────────────────
print_summary() {
    echo
    echo -e "${GREEN}${BOLD}═══════════════════════════════════════════════════════════════${NC}"
    echo -e "${GREEN}${BOLD}  Установка завершена!${NC}"
    echo -e "${GREEN}${BOLD}═══════════════════════════════════════════════════════════════${NC}"
    echo
    echo -e "Установленные компоненты:"
    echo -e "  ${BLUE}bspwm${NC}      — оконный менеджер (tiling)"
    echo -e "  ${BLUE}sxhkd${NC}      — горячие клавиши"
    echo -e "  ${BLUE}polybar${NC}    — панель с тач-кнопками"
    echo -e "  ${BLUE}rofi${NC}       — лаунчер (тач-тема)"
    echo -e "  ${BLUE}picom${NC}      — композитор"
    echo -e "  ${BLUE}onboard${NC}    — экранная клавиатура"

    case "$GESTURES" in
        libinput) echo -e "  ${BLUE}libinput-gestures${NC} — жесты тача" ;;
        fusuma)   echo -e "  ${BLUE}fusuma${NC}   — жесты тача" ;;
        none)     echo -e "  ${YELLOW}жесты не установлены${NC}" ;;
    esac

    echo -e "  ${BLUE}$TERMINAL${NC}  — терминал"

    echo
    echo -e "${CYAN}Запуск:${NC}"
    echo -e "  ${BOLD}startx${NC}"
    echo
    echo -e "${CYAN}Горячие клавиши:${NC}"
    echo -e "  Super+Enter   — терминал"
    echo -e "  Super+d       — rofi (лаунчер)"
    echo -e "  Super+1..0     — переключение рабочих столов"
    echo -e "  Super+hjkl     — фокус между окнами"
    echo -e "  Super+Shift+c — закрыть окно"
    echo -e "  Super+Space   — полноэкранный режим"
    echo -e "  Super+Esc     — перезагрузка sxhkd"
    echo -e "  Super+Shift+R — перезапуск bspwm"
    echo -e "  Super+Shift+Q — выход из bspwm"
    echo -e "  Super+Shift+K — экранная клавиатура"

    if [[ "$GESTURES" != "none" ]]; then
        echo
        echo -e "${CYAN}Жесты:${NC}"
        echo -e "  3 пальца вверх/вниз   — следующий/предыдущий стол"
        echo -e "  3 пальца влево/вправо — фокус на соседнее окно"
        echo -e "  4 пальца влево/вправо — перенос окна на соседний стол"
        echo -e "  Pinch внутрь          — закрыть окно"
        echo -e "  Pinch наружу          — запустить rofi"
    fi

    if [[ "$AUTO_LOGIN" == true ]]; then
        echo
        echo -e "${CYAN}Автологин${NC} включён — после перезагрузки X запустится автоматически."
    else
        echo
        echo -e "${YELLOW}Перезагрузитесь${NC}, затем запустите ${BOLD}startx${NC}."
    fi

    echo
    echo -e "${CYAN}Конфиги:${NC}"
    echo -e "  ~/.config/bspwm/bspwmrc"
    echo -e "  ~/.config/sxhkd/sxhkdrc"
    echo -e "  ~/.config/polybar/config.ini"
    echo -e "  ~/.config/rofi/touch.rasi"
    case "$GESTURES" in
        libinput) echo -e "  ~/.config/libinput-gestures.conf" ;;
        fusuma)   echo -e "  ~/.config/fusuma/config.yml" ;;
    esac

    echo
    echo -e "${CYAN}Логи:${NC}"
    echo -e "  bspwm:    bspc wm --report"
    echo -e "  polybar:  /tmp/polybar-example.log"
    echo -e "  gestures: libinput-gestures-setup status  (или journalctl для fusuma)"
    echo
}

# ─── Главный поток ─────────────────────────────────────────────────────────────
main() {
    echo -e "${GREEN}${BOLD}"
    echo "╔═══════════════════════════════════════════════════════════╗"
    echo "║     bspwm + polybar + rofi + touch  —  установщик        ║"
    echo "╚═══════════════════════════════════════════════════════════╝"
    echo -e "${NC}"

    check_os
    check_not_root
    check_sudo
    ask_questions

    if [[ "$BUILD_ISO" == true ]]; then
        # Сначала ставим всё на текущую систему, потом собираем ISO
        install_pacman_packages
        install_yay
        install_gestures
    else
        install_pacman_packages
        install_yay
        install_gestures
    fi

    setup_touch_permissions
    setup_xorg_touch
    create_dirs

    write_bspwmrc
    write_sxhkdrc
    write_polybar
    write_rofi
    write_power_menu
    write_gestures_config
    write_xinitrc

    setup_autologin

    if [[ "$BUILD_ISO" == true ]]; then
        build_iso
    fi

    print_summary

    echo -e "${GREEN}Готово!${NC}"
}

main "$@"
