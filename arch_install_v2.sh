#!/usr/bin/env bash
set -euo pipefail

# ============================================================
#  Arch Linux Installer v2 — Dynamic Edition
#  bspwm + polybar + rofi + Flatpak + dynamic package selection
# ============================================================

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BLUE='\033[0;34m'; CYAN='\033[0;36m'; MAGENTA='\033[1;35m'
BOLD='\033[1m'; NC='\033[0m'

banner() {
    echo -e "${CYAN}"
    echo "╔═══════════════════════════════════════════════════════════╗"
    echo "║   Arch Linux Installer v2 — Dynamic Selection Edition     ║"
    echo "║   bspwm · polybar · rofi · Flatpak · AUR · dynamic menus  ║"
    echo "╚═══════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

info()  { echo -e "${GREEN}[+]${NC} $1"; }
warn()  { echo -e "${YELLOW}[!]${NC} $1"; }
error() { echo -e "${RED}[ERROR]${NC} $1"; }
hr()    { echo -e "${BLUE}──────────────────────────────────────────────────────────${NC}"; }

# ── Проверка root ────────────────────────────────────────────
if [[ $EUID -ne 0 ]]; then
    error "Запустите от root: sudo bash $0"
    exit 1
fi

# ── Проверка UEFI ─────────────────────────────────────────────
UEFI=true
if [[ ! -d /sys/firmware/efi ]]; then
    warn "Legacy/BIOS режим. Скрипт лучше работает с UEFI."
    read -p "Продолжить в Legacy? (y/N): " ans
    [[ "$ans" =~ ^[Yy]$ ]] || exit 1
    UEFI=false
fi

banner

# ── Интернет ──────────────────────────────────────────────────
if ! ping -c1 -W2 archlinux.org &>/dev/null; then
    warn "Нет соединения. Настройте вручную:"
    echo "  iwctl        — Wi-Fi (Intel)"
    echo "  nmcli        — NetworkManager"
    echo "  dhcpcd       — Ethernet"
    read -p "Нажмите Enter после подключения..." _
fi

timedatectl set-ntp true
info "Часы синхронизированы."

# ══════════════════════════════════════════════════════════════
#  УТИЛИТЫ ДЛЯ ДИНАМИЧЕСКИХ МЕНЮ
# ══════════════════════════════════════════════════════════════

# Одинарный выбор из массива
# Использование: single_select "Заголовок" "item1|desc1" "item2|desc2" ...
# Вывод: SELECTED_ITEM
single_select() {
    local title="$1"; shift
    local items=("$@")
    local count=${#items[@]}
    
    while true; do
        echo -e "\n${BOLD}${title}${NC}"
        hr
        for i in "${!items[@]}"; do
            local item="${items[$i]}"
            local name="${item%%|*}"
            local desc="${item#*|}"
            printf "  ${CYAN}%2d)${NC} ${BOLD}%-22s${NC} %s\n" "$((i+1))" "$name" "$desc"
        done
        echo ""
        read -p "Выбор [1-$count]: " choice
        if [[ "$choice" =~ ^[0-9]+$ ]] && (( choice >= 1 && choice <= count )); then
            SELECTED_ITEM="${items[$((choice-1))]}"
            SELECTED_ITEM="${SELECTED_ITEM%%|*}"
            return 0
        fi
        warn "Неверный выбор."
    done
}

# Множественный выбор (чекбоксы, стрелки)
# Использование: multi_select "Заголовок" "item|desc|on" "item|desc|" ...
# Вывод: SELECTED_ITEMS (массив)
declare -a SELECTED_ITEMS=()

multi_select() {
    local title="$1"; shift
    local items=("$@")
    local count=${#items[@]}
    local -a states=()
    
    for item in "${items[@]}"; do
        IFS='|' read -ra parts <<< "$item"
        local last="${parts[-1]}"
        if [[ "$last" == "on" ]]; then
            states+=("on")
        else
            states+=("off")
        fi
    done
    
    local cursor=0
    
    while true; do
        echo -e "\n${BOLD}${title}${NC}"
        echo "  ${MAGENTA}↑/↓${NC} навигация · ${MAGENTA}Space${NC} переключить · ${MAGENTA}Enter${NC} подтвердить · ${MAGENTA}a${NC} всё · ${MAGENTA}q${NC} отменить"
        hr
        
        for i in "${!items[@]}"; do
            local item="${items[$i]}"
            IFS='|' read -ra parts <<< "$item"
            local name="${parts[0]}"
            local desc="${parts[1]}"
            local mark
            if [[ "${states[$i]}" == "on" ]]; then
                mark="${GREEN}[*]${NC}"
            else
                mark="${RED}[ ]${NC}"
            fi
            if [[ $i -eq $cursor ]]; then
                printf "  ${BOLD}${YELLOW}>${NC} ${mark} ${BOLD}%-24s${NC} %s\n" "$name" "$desc"
            else
                printf "    ${mark} %-24s %s\n" "$name" "$desc"
            fi
        done
        
        read -rsn1 key
        case "$key" in
            A|k)
                (( cursor-- )); (( cursor < 0 )) && cursor=$((count-1)) ;;
            B|j)
                (( cursor++ )); (( cursor >= count )) && cursor=0 ;;
            ' ')
                if [[ "${states[$cursor]}" == "on" ]]; then
                    states[$cursor]="off"
                else
                    states[$cursor]="on"
                fi ;;
            a|A)
                for i in "${!states[@]}"; do states[$i]="on"; done ;;
            q|Q)
                SELECTED_ITEMS=()
                return 1 ;;
            '')
                SELECTED_ITEMS=()
                for i in "${!items[@]}"; do
                    if [[ "${states[$i]}" == "on" ]]; then
                        IFS='|' read -ra parts <<< "${items[$i]}"
                        SELECTED_ITEMS+=("${parts[0]}")
                    fi
                done
                return 0 ;;
        esac
        
        echo -en "\033[$((count+4))A"
    done
}

# ══════════════════════════════════════════════════════════════
#  1. ВЫБОР ДИСКА
# ══════════════════════════════════════════════════════════════
echo -e "\n${BOLD}=== Выбор диска ===${NC}"
lsblk -do NAME,SIZE,MODEL,TRAN
echo ""
read -p "Диск (sda / nvme0n1 / ...): " DISK
DISK="/dev/${DISK#/dev/}"

[[ ! -b "$DISK" ]] && { error "Диск $DISK не найден."; exit 1; }

if [[ "$DISK" =~ nvme ]]; then
    PP="${DISK}p"
else
    PP="$DISK"
fi

echo -e "\n${RED}ВНИМАНИЕ: Все данные на $DISK будут уничтожены!${NC}"
read -p "Напишите YES для подтверждения: " CONFIRM
[[ "$CONFIRM" == "YES" ]] || { echo "Отмена."; exit 0; }

# ══════════════════════════════════════════════════════════════
#  2. РАЗМЕТКА И ФОРМАТИРОВАНИЕ
# ══════════════════════════════════════════════════════════════
echo -e "\n${BOLD}=== Разметка диска ===${NC}"

single_select "Схема разметки" \
    "auto-3|EFI (512M) + Swap + Root (остальное)|on" \
    "auto-4|EFI + Swap + Root + отдельный Home" \
    "manual|Ручная разметка через cfdisk"

PART_SCHEME="$SELECTED_ITEM"

wipefs -af "$DISK"
sgdisk -Z "$DISK"
info "Диск очищён."

if [[ "$PART_SCHEME" == "auto-3" ]]; then
    read -p "Размер swap в ГБ (например 8): " SWAP_GB
    sgdisk -n 1:0:+512M           -t 1:ef00 -c 1:"EFI"  "$DISK"
    sgdisk -n 2:0:+${SWAP_GB}G     -t 2:8200 -c 2:"SWAP" "$DISK"
    sgdisk -n 3:0:0               -t 3:8300 -c 3:"ROOT" "$DISK"
    EFI_PART="${PP}1"; SWAP_PART="${PP}2"; ROOT_PART="${PP}3"; HOME_PART=""

elif [[ "$PART_SCHEME" == "auto-4" ]]; then
    read -p "Размер swap в ГБ: " SWAP_GB
    read -p "Размер root в ГБ: " ROOT_GB
    sgdisk -n 1:0:+512M            -t 1:ef00 -c 1:"EFI"  "$DISK"
    sgdisk -n 2:0:+${SWAP_GB}G      -t 2:8200 -c 2:"SWAP" "$DISK"
    sgdisk -n 3:0:+${ROOT_GB}G      -t 3:8300 -c 3:"ROOT" "$DISK"
    sgdisk -n 4:0:0                 -t 4:8300 -c 4:"HOME" "$DISK"
    EFI_PART="${PP}1"; SWAP_PART="${PP}2"; ROOT_PART="${PP}3"; HOME_PART="${PP}4"

elif [[ "$PART_SCHEME" == "manual" ]]; then
    cfdisk "$DISK"
    echo "Укажите партиции:"
    read -p "EFI: " EFI_PART
    read -p "Swap: " SWAP_PART
    read -p "Root: " ROOT_PART
    read -p "Home (пусто если нет): " HOME_PART
fi

# ── Выбор ФС ──
single_select "Файловая система" \
    "ext4|Надёжная, проверенная|on" \
    "btrfs|Снапшоты, компрессия zstd" \
    "xfs|Быстрая, масштабируемая"

ROOT_FS="$SELECTED_ITEM"

# ── Форматирование ──
mkfs.fat -F32 "$EFI_PART"
info "EFI: $EFI_PART (FAT32)"

mkswap "$SWAP_PART"
swapon "$SWAP_PART"
info "Swap: $SWAP_PART"

case "$ROOT_FS" in
    ext4)  mkfs.ext4 -F "$ROOT_PART" ;;
    btrfs) mkfs.btrfs -f "$ROOT_PART" ;;
    xfs)   mkfs.xfs -f "$ROOT_PART" ;;
esac
info "Root: $ROOT_PART ($ROOT_FS)"

if [[ -n "$HOME_PART" ]]; then
    case "$ROOT_FS" in
        ext4)  mkfs.ext4 -F "$HOME_PART" ;;
        btrfs) mkfs.btrfs -f "$HOME_PART" ;;
        xfs)   mkfs.xfs -f "$HOME_PART" ;;
    esac
    info "Home: $HOME_PART ($ROOT_FS)"
fi

# ── Монтирование ──
mount "$ROOT_PART" /mnt

if [[ "$ROOT_FS" == "btrfs" ]]; then
    btrfs subvolume create /mnt/@ 2>/dev/null || true
    btrfs subvolume create /mnt/@home 2>/dev/null || true
    umount /mnt
    mount -o subvol=@,compress=zstd:1 "$ROOT_PART" /mnt
    mkdir -p /mnt/home
    if [[ -n "$HOME_PART" ]]; then
        mount "$HOME_PART" /mnt/home
    else
        mount -o subvol=@home,compress=zstd:1 "$ROOT_PART" /mnt/home
    fi
    info "Btrfs subvolumes: @ и @home"
else
    [[ -n "$HOME_PART" ]] && { mkdir -p /mnt/home; mount "$HOME_PART" /mnt/home; }
fi

mkdir -p /mnt/boot/efi
mount "$EFI_PART" /mnt/boot/efi
info "Разделы смонтированы."

# ══════════════════════════════════════════════════════════════
#  3. ДИНАМИЧЕСКИЙ ВЫБОР ЯДРА
# ══════════════════════════════════════════════════════════════
echo -e "\n${BOLD}=== Выбор ядра ===${NC}"

# Динамическая проверка доступности ядер в репозиториях
REPO_KERNELS=()
for k in linux linux-lts linux-zen linux-hardened linux-rt linux-rt-lts; do
    if pacman -Si "$k" &>/dev/null 2>&1; then
        REPO_KERNELS+=("$k")
    fi
done
# Fallback если pacman -Si не сработал на live USB
[[ ${#REPO_KERNELS[@]} -eq 0 ]] && REPO_KERNELS=(linux linux-lts linux-zen linux-hardened)

declare -A KERNEL_DESC=(
    [linux]="Стандартное ядро Arch (стабильное)"
    [linux-lts]="Long-term support (максимальная стабильность)"
    [linux-zen]="Оптимизировано для desktop/игр"
    [linux-hardened]="Усиленная безопасность"
    [linux-rt]="Real-time (аудио/видео продакшн)"
    [linux-rt-lts]="Real-time LTS"
    [linux-ck]="Оптимизация под CPU (AUR)"
    [linux-tkg]="TKG gaming kernel (AUR)"
    [linux-git]="Bleeding-edge из git (AUR)"
    [linux-xanmod]="XanMod — производительность (AUR)"
    [linux-lqx]="Liquorix — desktop/игры (AUR)"
)

KERNEL_ITEMS=()
for k in "${REPO_KERNELS[@]}"; do
    desc="${KERNEL_DESC[$k]:-Доступно в репозитории}"
    preselect=""
    [[ "$k" == "linux" ]] && preselect="|on"
    KERNEL_ITEMS+=("$k|$desc$preselect")
done

# AUR ядра
for k in linux-ck linux-tkg linux-git linux-xanmod linux-lqx; do
    desc="${KERNEL_DESC[$k]}"
    KERNEL_ITEMS+=("$k|$desc")
done

multi_select "Выберите ядра (можно несколько)" "${KERNEL_ITEMS[@]}"
SELECTED_KERNELS=("${SELECTED_ITEMS[@]}")

REPO_KERNEL_PKGS=""
AUR_KERNEL_PKGS=""
for k in "${SELECTED_KERNELS[@]}"; do
    case "$k" in
        linux|linux-lts|linux-zen|linux-hardened|linux-rt|linux-rt-lts)
            REPO_KERNEL_PKGS="$REPO_KERNEL_PKGS $k ${k}-headers"
            ;;
        *)
            AUR_KERNEL_PKGS="$AUR_KERNEL_PKGS $k ${k}-headers"
            ;;
    esac
done

# Если только AUR-ядро — ставим linux как fallback
if [[ -z "$REPO_KERNEL_PKGS" && -n "$AUR_KERNEL_PKGS" ]]; then
    REPO_KERNEL_PKGS="linux linux-headers"
    warn "AUR-ядро требует базового linux. linux будет установлен временно."
fi

info "Репо-ядра: ${REPO_KERNEL_PKGS:-нет}"
info "AUR-ядра: ${AUR_KERNEL_PKGS:-нет}"

# ══════════════════════════════════════════════════════════════
#  4. ВЫБОР АУДИОДРАВЕРА
# ══════════════════════════════════════════════════════════════
echo -e "\n${BOLD}=== Аудиодрайвер и звуковой стек ===${NC}"

single_select "Звуковой сервер" \
    "pipewire|Современный (WirePlumber, PulseAudio-совместимый)|on" \
    "pulseaudio|Классический PulseAudio" \
    "alsa-only|Только ALSA, без сервера"

AUDIO_SERVER="$SELECTED_ITEM"

single_select "Прошивки (firmware)" \
    "sof-firmware|Sound Open Firmware (ноутбуки)|on" \
    "alsa-firmware|Доп. прошивки для редких карт" \
    "none|Не устанавливать доп. прошивки"

AUDIO_FIRMWARE="$SELECTED_ITEM"

AUDIO_ITEMS=(
    "pavucontrol|Графический микшер громкости|on"
    "pamixer|CLI громкость (для polybar)|on"
    "pactl|Утилиты PulseAudio (pactl)|on"
    "pwvucontrol|Микшер PipeWire (новый)"
    "easyeffects|Звуковые эффекты PipeWire"
    "qpwgraph|Патч-бей для PipeWire"
    "helvum|Патч-бей для PipeWire (GTK)"
    "cadence|JACK-инструменты (музыка)"
)

multi_select "Дополнительные аудио-инструменты" "${AUDIO_ITEMS[@]}"
AUDIO_EXTRA=("${SELECTED_ITEMS[@]}")

# ══════════════════════════════════════════════════════════════
#  5. ВЫБОР ВИДЕОДРАЙВЕРА
# ══════════════════════════════════════════════════════════════
echo -e "\n${BOLD}=== Видеодрайвер ===${NC}"

single_select "Видеодрайвер" \
    "mesa|Открытый (Intel/AMD)|on" \
    "nvidia|Проприетарный NVIDIA"

VIDEO_DRIVER="$SELECTED_ITEM"

if [[ "$VIDEO_DRIVER" == "nvidia" ]]; then
    single_select "Вариант NVIDIA" \
        "nvidia|Стандартный (Turing+)|on" \
        "nvidia-lts|LTS-версия (для linux-lts)" \
        "nvidia-dkms|DKMS (под любое ядро)" \
        "nvidia-open|Открытый (Ada Lovelace+)"
    NVIDIA_VARIANT="$SELECTED_ITEM"
else
    NVIDIA_VARIANT=""
fi

# ══════════════════════════════════════════════════════════════
#  6. WM И КОМПОНЕНТЫ
# ══════════════════════════════════════════════════════════════
echo -e "\n${BOLD}=== Окружение рабочего стола ===${NC}"

WM_ITEMS=(
    "bspwm|Тайловый WM (основной)|on"
    "sxhkd|Горячие клавиши|on"
    "polybar|Панель/статус-бар|on"
    "rofi|Лаунчер приложений|on"
    "picom|Композитор (тени, прозрачность)|on"
    "dunst|Лёгкие уведомления|on"
    "feh|Управление обоями|on"
    "flameshot|Скриншоты с аннотацией|on"
    "xdotool|Автоматизация X11|on"
    "i3lock|Блокировка экрана|on"
    "xsecurelock|Безопасная блокировка|on"
    "wmctrl|Управление окнами из CLI"
    "unclutter|Скрытие курсора при простое"
)

multi_select "Компоненты WM" "${WM_ITEMS[@]}"
WM_PKGS=("${SELECTED_ITEMS[@]}")

# ══════════════════════════════════════════════════════════════
#  7. ТЕРМИНАЛ
# ══════════════════════════════════════════════════════════════
echo -e "\n${BOLD}=== Терминал ===${NC}"

TERM_ITEMS=(
    "alacritty|GPU-ускоренный, TOML-конфиг|on"
    "kitty|GPU, изображения, ligatures"
    "foot|Wayland-ориентированный, лёгкий"
    "wezterm|Мультиплексор, Lua-конфиг"
    "st|suckless terminal (компиляция)"
    "terminator|Мульти-оконный, GTK"
    "tilix|Тайловый терминал, GTK"
    "xterm|Классический X-терминал"
)

multi_select "Терминалы" "${TERM_ITEMS[@]}"
TERM_PKGS=("${SELECTED_ITEMS[@]}")

# ══════════════════════════════════════════════════════════════
#  8. ФАЙЛОВЫЕ МЕНЕДЖЕРЫ
# ══════════════════════════════════════════════════════════════
echo -e "\n${BOLD}=== Файловые менеджеры ===${NC}"

FM_ITEMS=(
    "ranger|Консольный, Python, превью|on"
    "lf|Консольный, Go, быстрый"
    "nnn|Консольный, C, минимальный"
    "mc|Midnight Commander"
    "yazi|Консольный, Rust (AUR)"
    "thunar|GUI, XFCE"
    "pcmanfm|GUI, LXDE"
    "nautilus|GUI, GNOME"
    "dolphin|GUI, KDE"
    "nemo|GUI, Cinnamon"
)

multi_select "Файловые менеджеры" "${FM_ITEMS[@]}"
FM_PKGS=("${SELECTED_ITEMS[@]}")

# ══════════════════════════════════════════════════════════════
#  9. БРАУЗЕРЫ
# ══════════════════════════════════════════════════════════════
echo -e "\n${BOLD}=== Браузеры ===${NC}"

BROWSER_ITEMS=(
    "firefox|Mozilla Firefox (репо)|on"
    "chromium|Chromium (репо)"
    "qutebrowser|Клавиатурный браузер (репо)"
    "falkon|Лёгкий KDE-браузер (репо)"
    "brave-bin|Brave (AUR)"
    "google-chrome|Google Chrome (AUR)"
    "vivaldi|Vivaldi (AUR)"
    "zen-browser-bin|Zen Browser (AUR)"
    "librewolf-bin|LibreWolf (AUR)"
    "floorp|Floorp (AUR)"
)

multi_select "Браузеры" "${BROWSER_ITEMS[@]}"
BROWSER_PKGS=("${SELECTED_ITEMS[@]}")

# ══════════════════════════════════════════════════════════════
#  10. РЕДАКТОРЫ И IDE
# ══════════════════════════════════════════════════════════════
echo -e "\n${BOLD}=== Редакторы и IDE ===${NC}"

EDITOR_ITEMS=(
    "neovim|Современный Vim (Lua)|on"
    "vim|Классический Vim"
    "nano|Простой консольный|on"
    "emacs|GNU Emacs (консоль/GUI)"
    "micro|Современный консольный"
    "helix|Консольный, builtin LSP"
    "kakoune|m-selection редактор (AUR)"
    "code|VS Code — OSS (репо)"
    "code-bin|VS Code бинарный (AUR)"
    "vscodium-bin|VSCodium — без телеметрии (AUR)"
    "sublime-text-4|Sublime Text 4 (AUR)"
    "zed|Zed editor (AUR)"
)

multi_select "Редакторы и IDE" "${EDITOR_ITEMS[@]}"
EDITOR_PKGS=("${SELECTED_ITEMS[@]}")

# ══════════════════════════════════════════════════════════════
#  11. ПРИЛОЖЕНИЯ ПО КАТЕГОРИЯМ
# ══════════════════════════════════════════════════════════════

# ── Система ──
echo -e "\n${BOLD}=== Системные утилиты ===${NC}"
SYS_ITEMS=(
    "htop|Мониторинг процессов|on"
    "btop|Продвинутый TUI монитор|on"
    "neofetch|Информация о системе|on"
    "fastfetch|Быстрый neofetch|on"
    "lm_sensors|Датчики температуры|on"
    "smartmonttools|SMART мониторинг дисков"
    "gparted|Графический менеджер разделов"
    "timeshift|Снапшоты системы (btrfs)|on"
    "reflector|Обновление зеркал pacman"
    "pacman-contrib|Утилиты pacman|on"
    "paru|AUR-хелпер на Rust (AUR)"
)
multi_select "Системные утилиты" "${SYS_ITEMS[@]}"
SYS_PKGS=("${SELECTED_ITEMS[@]}")

# ── Сеть ──
echo -e "\n${BOLD}=== Сеть и загрузчики ===${NC}"
NET_ITEMS=(
    "networkmanager|NetworkManager|on"
    "network-manager-applet|Апплет для панели|on"
    "wireless_tools|Утилиты Wi-Fi|on"
    "wpa_supplicant|WPA аутентификация|on"
    "bluez|Bluetooth стек|on"
    "bluez-utils|Bluetooth CLI|on"
    "blueman|Bluetooth-апплет|on"
    "wireguard-tools|WireGuard VPN"
    "openvpn|OpenVPN клиент"
    "qbittorrent|qBittorrent"
    "aria2|Многопоточный загрузчик|on"
    "yt-dlp|YouTube/видео загрузчик|on"
)
multi_select "Сеть и загрузчики" "${NET_ITEMS[@]}"
NET_PKGS=("${SELECTED_ITEMS[@]}")

# ── Мультимедиа ──
echo -e "\n${BOLD}=== Мультимедиа и графика ===${NC}"
MEDIA_ITEMS=(
    "mpv|Медиаплеер минималистичный|on"
    "vlc|VLC медиаплеер"
    "celluloid|GTK-обёртка для mpv"
    "feh|Просмотрщик изображений|on"
    "nsxiv|Лёгкий просмотрщик (sxiv fork)"
    "gimp|Графический редактор"
    "inkscape|Векторный редактор SVG"
    "krita|Цифровая живопись"
    "blender|3D моделирование"
    "obs-studio|Запись/стриминг экрана"
    "ffmpeg|Видеокодек/конвертер|on"
    "imagemagick|Работа с изображениями|on"
    "audacity|Аудиоредактор"
    "kdenlive|Видеоредактор"
)
multi_select "Мультимедиа и графика" "${MEDIA_ITEMS[@]}"
MEDIA_PKGS=("${SELECTED_ITEMS[@]}")

# ── Офис ──
echo -e "\n${BOLD}=== Офис и документы ===${NC}"
OFFICE_ITEMS=(
    "libreoffice-fresh|LibreOffice (fresh)|on"
    "libreoffice-stable|LibreOffice (stable)"
    "zathura|Лёгкий PDF (vim-клавиши)|on"
    "zathura-pdf-mupdf|PDF плагин для zathura|on"
    "evince|GNOME Document Viewer"
    "calibre|Менеджер электронных книг"
    "pandoc|Конвертер документов|on"
    "texlive-core|LaTeX (базовый)"
)
multi_select "Офис и документы" "${OFFICE_ITEMS[@]}"
OFFICE_PKGS=("${SELECTED_ITEMS[@]}")

# ── Разработка ──
echo -e "\n${BOLD}=== Разработка ===${NC}"
DEV_ITEMS=(
    "git|Контроль версий|on"
    "base-devel|Инструменты сборки|on"
    "cmake|Сборка C/C++"
    "make|GNU Make|on"
    "gcc|GCC компилятор|on"
    "clang|LLVM/Clang"
    "rustup|Rust toolchain|on"
    "go|Go language"
    "python|Python 3|on"
    "python-pip|pip для Python|on"
    "nodejs|Node.js|on"
    "npm|npm|on"
    "docker|Контейнеризация"
    "docker-compose|Docker Compose"
    "jq|JSON процессор|on"
    "fzf|Fuzzy finder|on"
    "ripgrep|Быстрый grep (rg)|on"
    "fd|Быстрый find|on"
    "bat|cat с подсветкой|on"
    "eza|ls с git (AUR)"
    "lazygit|TUI для git|on"
    "gh|GitHub CLI|on"
)
multi_select "Разработка и инструменты" "${DEV_ITEMS[@]}"
DEV_PKGS=("${SELECTED_ITEMS[@]}")

# ── Шрифты ──
echo -e "\n${BOLD}=== Шрифты ===${NC}"
FONT_ITEMS=(
    "ttf-jetbrains-mono|JetBrains Mono|on"
    "ttf-nerd-fonts-symbols-mono|Nerd Font символы|on"
    "nerd-fonts-jetbrains-mono|JetBrains Mono Nerd (AUR)"
    "nerd-fonts-fira-code|Fira Code Nerd (AUR)"
    "nerd-fonts-hack|Hack Nerd (AUR)"
    "ttf-fira-code|Fira Code (лигатуры)"
    "noto-fonts|Noto (Unicode)|on"
    "noto-fonts-cjk|Noto CJK (китайский/японский)"
    "noto-fonts-emoji|Noto Emoji|on"
    "ttf-liberation|Liberation (MS-совместимые)"
    "ttf-dejavu|DejaVu (классика)"
    "ttf-cascadia-code|Cascadia Code (MS)"
    "inter-font|Inter (современный sans)"
)
multi_select "Шрифты" "${FONT_ITEMS[@]}"
FONT_PKGS=("${SELECTED_ITEMS[@]}")

# ── Темы ──
echo -e "\n${BOLD}=== Темы и иконки ===${NC}"
THEME_ITEMS=(
    "papirus-icon-theme|Papirus иконки|on"
    "arc-gtk-theme|Arc GTK тема"
    "catppuccin-gtk-theme-macchiato|Catppuccin (AUR)"
    "kvantum|Kvantum (SVG темы Qt)"
    "qt5ct|Настройка Qt5|on"
    "qt6ct|Настройка Qt6"
    "lxappearance|Настройка GTK тем|on"
    "sddm|Дисплейный менеджер"
    "greetd|Минимальный DM"
    "ly|TUI display manager (AUR)"
)
multi_select "Темы и иконки" "${THEME_ITEMS[@]}"
THEME_PKGS=("${SELECTED_ITEMS[@]}")

# ── Мессенджеры ──
echo -e "\n${BOLD}=== Мессенджеры ===${NC}"
CHAT_ITEMS=(
    "telegram-desktop|Telegram (репо)|on"
    "discord|Discord (репо)"
    "discord_arch_electron|Discord на системном Electron (AUR)"
    "webcord-bin|Webcord (AUR)"
    "signal-desktop|Signal (AUR)"
    "slack-desktop|Slack (AUR)"
    "zoom|Zoom (AUR)"
    "weechat|IRC клиент (консоль)"
)
multi_select "Мессенджеры и общение" "${CHAT_ITEMS[@]}"
CHAT_PKGS=("${SELECTED_ITEMS[@]}")

# ══════════════════════════════════════════════════════════════
#  12. FLATPAK
# ══════════════════════════════════════════════════════════════
echo -e "\n${BOLD}=== Flatpak приложения ===${NC}"

FLATPAK_ITEMS=(
    "com.valvesoftware.Steam|Steam (игры)|on"
    "com.spotify.Client|Spotify|on"
    "org.mozilla.firefox|Firefox (Flatpak)"
    "org.chromium.Chromium|Chromium (Flatpak)"
    "com.brave.Browser|Brave (Flatpak)"
    "org.libreoffice.LibreOffice|LibreOffice|on"
    "org.gimp.GIMP|GIMP|on"
    "org.inkscape.Inkscape|Inkscape|on"
    "org.kde.krita|Krita|on"
    "org.blender.Blender|Blender|on"
    "org.obsproject.Studio|OBS Studio|on"
    "com.discordapp.Discord|Discord|on"
    "org.telegram.desktop|Telegram (Flatpak)|on"
    "org.signal.Signal|Signal|on"
    "org.zoom.Zoom|Zoom|on"
    "com.slack.Slack|Slack|on"
    "com.visualstudio.code|VS Code (Flatpak)"
    "com.vscodium.codium|VSCodium (Flatpak)"
    "com.transmissionbt.Transmission|Transmission|on"
    "org.qbittorrent.qBittorrent|qBittorrent|on"
    "org.audacityteam.Audacity|Audacity|on"
    "org.kde.kdenlive|Kdenlive|on"
    "md.obsidian.Obsidian|Obsidian|on"
    "com.github.tchx84.Flatseal|Flatseal (управление Flatpak)|on"
    "org.gnome.Calculator|Калькулятор GNOME"
    "org.gnome.FileRoller|Архиватор GNOME"
    "org.gnome.Evince|PDF просмотрщик"
    "org.kde.okular|Okular PDF (KDE)"
)

multi_select "Flatpak приложения" "${FLATPAK_ITEMS[@]}"
FLATPAK_PKGS=("${SELECTED_ITEMS[@]}")

# ══════════════════════════════════════════════════════════════
#  13. СИСТЕМНЫЕ ПАРАМЕТРЫ
# ══════════════════════════════════════════════════════════════
echo -e "\n${BOLD}=== Системные параметры ===${NC}"

read -p "Имя хоста: " HOSTNAME
read -p "Имя пользователя: " USERNAME
read -s -p "Пароль пользователя: " USERPASS; echo
read -s -p "Пароль root: " ROOTPASS; echo

single_select "Часовой пояс" \
    "Europe/Moscow|MSK UTC+3|on" \
    "Europe/Saratov|UTC+4" \
    "Europe/Samara|UTC+4" \
    "Europe/Yekaterinburg|UTC+5" \
    "Asia/Novosibirsk|UTC+7" \
    "Asia/Vladivostok|UTC+10" \
    "custom|Указать вручную"
TIMEZONE="$SELECTED_ITEM"
[[ "$TIMEZONE" == "custom" ]] && read -p "Часовой пояс: " TIMEZONE

single_select "Локаль" \
    "ru_RU.UTF-8|Русский|on" \
    "en_US.UTF-8|English" \
    "custom|Указать вручную"
LOCALE="$SELECTED_ITEM"
[[ "$LOCALE" == "custom" ]] && read -p "Локаль: " LOCALE

single_select "Переключение раскладки" \
    "win_space_toggle|Win+Space|on" \
    "caps_toggle|CapsLock" \
    "alt_shift_toggle|Alt+Shift" \
    "ctrl_shift_toggle|Ctrl+Shift"
KBD_TOGGLE="$SELECTED_ITEM"

# ══════════════════════════════════════════════════════════════
#  14. СБОРКА И КЛАССИФИКАЦИЯ ПАКЕТОВ
# ══════════════════════════════════════════════════════════════
REPO_PKGS=""
AUR_PKGS=""

# Список известных AUR-пакетов
AUR_LIST="linux-ck linux-tkg linux-git linux-xanmod linux-lqx linux-asahi \
    brave-bin google-chrome vivaldi zen-browser-bin librewolf-bin floorp waterfox \
    yazi kakoune code-bin vscodium-bin sublime-text-4 zed \
    paru eza catppuccin-gtk-theme-macchiato ly \
    discord_arch_electron webcord-bin signal-desktop slack-desktop zoom \
    lazygit timeshift"

add_pkg() {
    local pkg="$1"
    if echo "$AUR_LIST" | grep -qw "$pkg"; then
        AUR_PKGS="$AUR_PKGS $pkg"
    else
        REPO_PKGS="$REPO_PKGS $pkg"
    fi
}

# Базовые пакеты
REPO_PKGS="base base-devel linux-firmware btrfs-progs efibootmgr grub os-prober \
    mtools dosfstools ntfs-3g exfatprogs f2fs-tools \
    openssh sudo git curl wget rsync unzip zip p7zip \
    bash-completion man-db man-pages texinfo \
    xdg-user-dirs xdg-utils gvfs gvfs-mtp gvfs-smb \
    polkit-gnome xsecurelock brightnessctl playerctl \
    xorg xorg-xinit xorg-xrandr xorg-xsetroot xorg-xprop xorg-xinput xorg-xev \
    ttf-dejavu noto-fonts noto-fonts-emoji $REPO_KERNEL_PKGS"

# Аудио стек
case "$AUDIO_SERVER" in
    pipewire)  REPO_PKGS="$REPO_PKGS pipewire pipewire-pulse pipewire-alsa pipewire-jack wireplumber" ;;
    pulseaudio) REPO_PKGS="$REPO_PKGS pulseaudio pulseaudio-alsa pulseaudio-bluetooth pulseaudio-jack" ;;
    alsa-only) REPO_PKGS="$REPO_PKGS alsa-utils alsa-plugins" ;;
esac

# Прошивки
case "$AUDIO_FIRMWARE" in
    sof-firmware)  REPO_PKGS="$REPO_PKGS sof-firmware" ;;
    alsa-firmware) REPO_PKGS="$REPO_PKGS alsa-firmware" ;;
esac

# Видеодрайвер
if [[ "$VIDEO_DRIVER" == "mesa" ]]; then
    REPO_PKGS="$REPO_PKGS mesa mesa-utils lib32-mesa vulkan-radeon vulkan-intel \
        vulkan-mesa-layers libva-mesa-driver mesa-vdpau"
elif [[ "$VIDEO_DRIVER" == "nvidia" ]]; then
    REPO_PKGS="$REPO_PKGS $NVIDIA_VARIANT lib32-nvidia-utils nvidia-utils"
    [[ "$NVIDIA_VARIANT" == "nvidia-dkms" ]] && REPO_PKGS="$REPO_PKGS dkms"
fi

# Обработка категорий
for cat_name in WM TERM FM BROWSER EDITOR SYS NET MEDIA OFFICE DEV FONT THEME CHAT AUDIO_EXTRA; do
    declare -n cat_array="${cat_name}_PKGS"
    for pkg in "${cat_array[@]:-}"; do
        add_pkg "$pkg"
    done
done

# AUR-ядра
[[ -n "$AUR_KERNEL_PKGS" ]] && AUR_PKGS="$AUR_PKGS $AUR_KERNEL_PKGS"

# Дедупликация
REPO_PKGS=$(echo "$REPO_PKGS" | tr ' ' '\n' | sort -u | tr '\n' ' ' | xargs)
AUR_PKGS=$(echo "$AUR_PKGS" | tr ' ' '\n' | sort -u | tr '\n' ' ' | xargs)

info "Репо: $(echo "$REPO_PKGS" | wc -w) пакетов"
info "AUR: $(echo "$AUR_PKGS" | wc -w) пакетов"
info "Flatpak: ${#FLATPAK_PKGS[@]} приложений"

# ══════════════════════════════════════════════════════════════
#  15. PACSTRAP
# ══════════════════════════════════════════════════════════════
echo -e "\n${BOLD}=== Установка базовой системы ===${NC}"

if command -v reflector &>/dev/null; then
    info "Обновление зеркал..."
    reflector --country Russia --age 12 --protocol https --sort rate \
        --save /etc/pacman.d/mirrorlist || warn "Не удалось обновить зеркала."
fi

info "Запуск pacstrap..."
pacstrap /mnt $REPO_PKGS
info "Базовая система установлена."

genfstab -U /mnt >> /mnt/etc/fstab
info "fstab сгенерирован."

# ══════════════════════════════════════════════════════════════
#  16. НАСТРОЙКА В CHROOT
# ══════════════════════════════════════════════════════════════
echo -e "\n${BOLD}=== Настройка системы ===${NC}"

case "$KBD_TOGGLE" in
    win_space_toggle)  KBD_OPTS="grp:win_space_toggle" ;;
    caps_toggle)       KBD_OPTS="grp:caps_toggle" ;;
    alt_shift_toggle)   KBD_OPTS="grp:alt_shift_toggle" ;;
    ctrl_shift_toggle)  KBD_OPTS="grp:ctrl_shift_toggle" ;;
    *)                  KBD_OPTS="grp:win_space_toggle" ;;
esac

arch-chroot /mnt /bin/bash -e <<CHROOT_EOF
ln -sf /usr/share/zoneinfo/${TIMEZONE} /etc/localtime
hwclock --systohc

sed -i 's/^#${LOCALE}/${LOCALE}/' /etc/locale.gen
sed -i 's/^#en_US.UTF-8/en_US.UTF-8/' /etc/locale.gen
locale-gen
echo "LANG=${LOCALE}" > /etc/locale.conf

if echo "${LOCALE}" | grep -q "ru_RU"; then
    echo "KEYMAP=ru" > /etc/vconsole.conf
    echo "FONT=cyr-sun16" >> /etc/vconsole.conf
fi

echo "${HOSTNAME}" > /etc/hostname
cat > /etc/hosts <<HOSTS
127.0.0.1   localhost
::1         localhost
127.0.1.1   ${HOSTNAME}.localdomain ${HOSTNAME}
HOSTS

echo "root:${ROOTPASS}" | chpasswd
useradd -m -G wheel,audio,video,storage,optical,network,power,lp,lpadmin -s /bin/bash ${USERNAME}
echo "${USERNAME}:${USERPASS}" | chpasswd

sed -i 's/^# %wheel ALL=(ALL:ALL) ALL/%wheel ALL=(ALL:ALL) ALL/' /etc/sudoers

systemctl enable NetworkManager
systemctl enable bluetooth 2>/dev/null || true
systemctl enable systemd-timesyncd

if [[ "${UEFI}" == "true" ]]; then
    grub-install --target=x86_64-efi --efi-directory=/boot/efi --bootloader-id=GRUB
else
    grub-install --target=i386-pc ${DISK}
fi
sed -i 's/^#GRUB_DISABLE_OS_PROBER=false/GRUB_DISABLE_OS_PROBER=false/' /etc/default/grub

# NVIDIA
if [[ "${VIDEO_DRIVER}" == "nvidia" ]]; then
    sed -i 's/^MODULES=()/MODULES=(nvidia nvidia_modeset nvidia_uvm nvidia_drm)/' /etc/mkinitcpio.conf
    sed -i 's/^GRUB_CMDLINE_LINUX=""/GRUB_CMDLINE_LINUX="nvidia-drm.modeset=1"/' /etc/default/grub
fi

# btrfs
if [[ "${ROOT_FS}" == "btrfs" ]]; then
    sed -i 's/^MODULES=()/MODULES=(btrfs)/' /etc/mkinitcpio.conf
fi

grub-mkconfig -o /boot/grub/grub.cfg
mkinitcpio -P
CHROOT_EOF

info "Системная настройка завершена."

# ══════════════════════════════════════════════════════════════
#  17. ПРЕДНАСТРОЙКА WM
# ══════════════════════════════════════════════════════════════
echo -e "\n${BOLD}=== Преднастройка WM ===${NC}"

UH="/mnt/home/${USERNAME}"
CD="$UH/.config"
mkdir -p "$CD/bspwm" "$CD/sxhkd" "$CD/polybar" "$CD/rofi" \
         "$CD/picom" "$CD/dunst" "$CD/autostart" \
         "$UH/.local/bin" "$UH/Pictures/wallpapers"

# Определение дефолтных приложений
DEFAULT_TERM="alacritty"
for t in "${TERM_PKGS[@]:-alacritty}"; do
    case "$t" in alacritty|kitty|foot|wezterm|terminator|tilix|xterm|st) DEFAULT_TERM="$t"; break ;; esac
done

DEFAULT_FM="ranger"
for f in "${FM_PKGS[@]:-ranger}"; do
    case "$f" in ranger|lf|nnn|mc|yazi|thunar|pcmanfm|nautilus|dolphin|nemo) DEFAULT_FM="$f"; break ;; esac
done

DEFAULT_BROWSER="firefox"
for b in "${BROWSER_PKGS[@]:-firefox}"; do
    case "$b" in
        firefox|chromium|qutebrowser|falkon) DEFAULT_BROWSER="$b"; break ;;
        brave-bin) DEFAULT_BROWSER="brave"; break ;;
        google-chrome) DEFAULT_BROWSER="google-chrome-stable"; break ;;
        vivaldi) DEFAULT_BROWSER="vivaldi-stable"; break ;;
        zen-browser-bin) DEFAULT_BROWSER="zen-browser"; break ;;
        librewolf-bin) DEFAULT_BROWSER="librewolf"; break ;;
        floorp) DEFAULT_BROWSER="floorp"; break ;;
    esac
done

DEFAULT_EDITOR="nvim"
for e in "${EDITOR_PKGS[@]:-neovim}"; do
    case "$e" in
        neovim) DEFAULT_EDITOR="nvim"; break ;;
        vim) DEFAULT_EDITOR="vim"; break ;;
        nano) DEFAULT_EDITOR="nano"; break ;;
        emacs) DEFAULT_EDITOR="emacs"; break ;;
        micro) DEFAULT_EDITOR="micro"; break ;;
        helix) DEFAULT_EDITOR="hx"; break ;;
        kakoune) DEFAULT_EDITOR="kak"; break ;;
        code|code-bin) DEFAULT_EDITOR="code"; break ;;
        vscodium-bin) DEFAULT_EDITOR="codium"; break ;;
        sublime-text-4) DEFAULT_EDITOR="subl"; break ;;
        zed) DEFAULT_EDITOR="zeditor"; break ;;
    esac
done

# bspwmrc
cat > "$CD/bspwm/bspwmrc" <<BSPWMRC
#!/usr/bin/env bash
\$HOME/.config/polybar/launch.sh &
picom --config \$HOME/.config/picom/picom.conf &
dunst -config \$HOME/.config/dunst/dunstrc &
dex -a -s \$HOME/.config/autostart &
nm-applet &
/usr/lib/polkit-gnome/polkit-gnome-authentication-agent-1 &
setxkbmap -layout us,ru -option grp:${KBD_OPTS} &
xsetroot -cursor_name left_ptr &
feh --bg-scale \$HOME/Pictures/wallpapers/wallpaper.jpg &
flameshot &

bspc rule -r "*"
bspc rule -a Screenkey manage=off
bspc rule -a zathura state=tiled
bspc rule -a Pavucontrol state=floating

bspc monitor -d I II III IV V VI VII VIII IX X

bspc config border_width          2
bspc config window_gap            8
bspc config top_padding           28
bspc config split_ratio           0.52
bspc config borderless_monocle    true
bspc config gapless_monocle       true
bspc config focus_follows_pointer true
bspc config pointer_modifier      mod4

bspc config normal_border_color   "#2e3440"
bspc config active_border_color   "#81a1c1"
bspc config focused_border_color  "#88c0d0"
bspc config presel_feedback_color "#5e81ac"
BSPWMRC
chmod +x "$CD/bspwm/bspwmrc"

# sxhkdrc
cat > "$CD/sxhkd/sxhkdrc" <<SXHKDRC
# Терминал
super + Return
    ${DEFAULT_TERM}

# Файловый менеджер
super + e
    ${DEFAULT_TERM} -e ${DEFAULT_FM}

# Rofi
super + d
    rofi -show drun -show-icons
super + Tab
    rofi -show window -show-icons

# Браузер
super + w
    ${DEFAULT_BROWSER}

# Редактор
super + n
    ${DEFAULT_TERM} -e ${DEFAULT_EDITOR}

# Окна
super + q
    bspc node -c
super + shift + q
    bspc node -k
super + f
    bspc node -t \~fullscreen
super + space
    bspc node -t \~floating

# Фокусировка
super + {h,j,k,l}
    bspc node -f {west,south,north,east}
super + {Left,Down,Up,Right}
    bspc node -f {west,south,north,east}

# Перемещение
super + shift + {h,j,k,l}
    bspc node -s {west,south,north,east}

# Рабочие столы
super + {1-9,0}
    bspc desktop -f {I,II,III,IV,V,VI,VII,VIII,IX,X}
super + shift + {1-9,0}
    bspc node -d {I,II,III,IV,V,VI,VII,VIII,IX,X}

# Предустановка
super + ctrl + {h,j,k,l}
    bspc node -p {west,south,north,east}
super + ctrl + space
    bspc node -p cancel

# Размер
super + alt + {h,j,k,l}
    bspc node -z {left -20 0, bottom 0 20, top 0 -20, right 20 0}

# Перезапуск
super + shift + r
    bspc wm -r && bspc quit
super + Escape
    pkill -USR1 -x sxhkd

# Блокировка
super + shift + x
    xsecurelock

# Громкость
XF86AudioRaiseVolume
    pamixer -i 5
XF86AudioLowerVolume
    pamixer -d 5
XF86AudioMute
    pamixer -t

# Яркость
XF86MonBrightnessUp
    brightnessctl set +5%
XF86MonBrightnessDown
    brightnessctl set 5%-

# Медиа
XF86AudioPlay
    playerctl play-pause
XF86AudioNext
    playerctl next
XF86AudioPrev
    playerctl previous

# Скриншот
Print
    flameshot gui
super + Print
    flameshot full -p ~/Pictures
SXHKDRC

# polybar
cat > "$CD/polybar/config.ini" <<'POLYBAR'
[colors]
background = #1a1b26
background-alt = #1f2335
foreground = #a9b1d6
primary = #7aa2f7
secondary = #bb9af7
alert = #f7768e
disabled = #565f89
green = #9ece6a
yellow = #e0af68

[bar/main]
width = 100%
height = 28pt
radius = 0
background = ${colors.background}
foreground = ${colors.foreground}
line-size = 3
border-size = 0pt
padding-left = 0
padding-right = 1
module-margin-left = 1
module-margin-right = 1
font-0 = "JetBrains Mono:size=10;2"
font-1 = "Symbols Nerd Font:size=12;2"
modules-left = xworkspaces xwindow
modules-center = date
modules-right = filesystem pulseaudio memory cpu temperature wlan eth battery tray
cursor-click = pointer
cursor-scroll = ns-resize
enable-ipc = true
tray-position = right
tray-padding = 2

[module/xworkspaces]
type = internal/xworkspaces
label-active = %name%
label-active-background = ${colors.primary}
label-active-foreground = ${colors.background}
label-active-padding = 1
label-occupied = %name%
label-occupied-foreground = ${colors.primary}
label-occupied-padding = 1
label-urgent = %name%
label-urgent-foreground = ${colors.alert}
label-urgent-padding = 1
label-empty = %name%
label-empty-foreground = ${colors.disabled}
label-empty-padding = 1

[module/xwindow]
type = internal/xwindow
label = %title:0:50:...%

[module/filesystem]
type = internal/fs
interval = 25
mount-0 = /
label-mounted = %{F#7aa2f7}DISK%{F-} %percentage_used%%
label-unmounted = DISK --

[module/pulseaudio]
type = internal/pulseaudio
format-volume-prefix = "VOL "
format-volume-prefix-foreground = ${colors.primary}
label-volume = %percentage%%
label-muted = MUTED
label-muted-foreground = ${colors.disabled}

[module/memory]
type = internal/memory
interval = 2
format-prefix = "RAM "
format-prefix-foreground = ${colors.primary}
label = %percentage_used:2%%

[module/cpu]
type = internal/cpu
interval = 2
format-prefix = "CPU "
format-prefix-foreground = ${colors.primary}
label = %percentage:2%%

[module/temperature]
type = internal/temperature
interval = 5
thermal-zone = 0
warn-temperature = 80
format-prefix = "TEMP "
format-prefix-foreground = ${colors.yellow}
label = %temperature-c%
label-warn-foreground = ${colors.alert}

[module/wlan]
type = internal/network
interface-type = wireless
interval = 3
label-connected = %{F#7aa2f7}WIFI%{F-} %essid:0:10%
label-disconnected = WIFI --

[module/eth]
type = internal/network
interface-type = wired
interval = 3
label-connected = %{F#7aa2f7}ETH%{F-} %local_ip%
label-disconnected = ETH --

[module/battery]
type = internal/battery
full-at = 99
low-at = 15
format-charging-prefix = "BAT+ "
format-charging-prefix-foreground = ${colors.green}
format-discharging-prefix = "BAT "
format-discharging-prefix-foreground = ${colors.primary}
label-charging = %percentage%%
label-discharging = %percentage%%
label-low = LOW %percentage%%
label-low-foreground = ${colors.alert}

[module/date]
type = internal/date
interval = 1
date = %H:%M
date-alt = %Y-%m-%d %H:%M:%S
label = %date%
label-foreground = ${colors.primary}

[module/tray]
type = internal/tray
format-margin = 8px

[settings]
screenchange-reload = true
POLYBAR

cat > "$CD/polybar/launch.sh" <<'POLYBAR_LAUNCH'
#!/usr/bin/env bash
killall -q polybar
while pgrep -u $UID -x polybar >/dev/null; do sleep 0.5; done
polybar main 2>&1 | tee -a /tmp/polybar.log &
disown
echo "Polybar started."
POLYBAR_LAUNCH
chmod +x "$CD/polybar/launch.sh"

# rofi
cat > "$CD/rofi/config.rasi" <<'ROFI'
configuration {
    modi: "drun,window,run,ssh";
    font: "JetBrains Mono 11";
    show-icons: true;
    icon-theme: "Papirus";
    terminal: "alacritty";
    drun-display-format: "{name}";
    window-format: "{w} · {c} · {t}";
    sidebar-mode: true;
    matching: "fuzzy";
    sort: true;
}

@theme "/dev/null"

* {
    background: #1a1b26;
    background-alt: #1f2335;
    foreground: #a9b1d6;
    selected: #7aa2f7;
    active: #9ece6a;
    urgent: #f7768e;
}

window {
    width: 38%;
    background-color: @background;
    border: 2px solid @selected;
    border-radius: 6;
    padding: 8;
}

inputbar {
    children: [ "prompt", "entry" ];
    background-color: @background-alt;
    text-color: @foreground;
    padding: 8;
    border-radius: 4;
}

prompt { text-color: @selected; padding: 0 8 0 0; }
entry { placeholder: "Search..."; placeholder-color: @foreground; text-color: @foreground; }

listview {
    background-color: transparent;
    lines: 12;
    columns: 1;
    cycle: true;
    dynamic: true;
    layout: vertical;
    scrollbar: true;
}

element { padding: 8; border-radius: 4; cursor: pointer; }
element selected { background-color: @selected; text-color: @background; }
element-icon { size: 24px; margin: 0 8 0 0; }

scrollbar { width: 4px; background-color: @background-alt; handle-color: @selected; border-radius: 2px; }
ROFI

# picom
cat > "$CD/picom/picom.conf" <<'PICOM'
backend = "glx";
vsync = true;
shadow = true;
shadow-radius = 7;
shadow-offset-x = -7;
shadow-offset-y = -7;
shadow-opacity = 0.5;
shadow-exclude = [
    "name = 'Notification'",
    "class_g = 'polybar'",
    "class_g = 'dunst'",
    "_GTK_FRAME_EXTENTS@:c"
];
fading = true;
fade-in-step = 0.03;
fade-out-step = 0.03;
fade-delta = 5;
inactive-opacity = 0.95;
active-opacity = 1.0;
frame-opacity = 1.0;
corner-radius = 6;
round-borders = 1;
opacity-rule = [
    "100:class_g = 'mpv'",
    "100:class_g = 'Alacritty'"
];
PICOM

# dunst
cat > "$CD/dunst/dunstrc" <<'DUNST'
[global]
    monitor = 0
    follow = mouse
    geometry = "350x5-10+40"
    indicate_hidden = yes
    shrink = no
    transparency = 0
    separator_height = 2
    padding = 8
    horizontal_padding = 8
    frame_width = 2
    frame_color = "#7aa2f7"
    separator_color = frame
    sort = yes
    idle_threshold = 120
    font = JetBrains Mono 10
    markup = full
    format = "<b>%s</b>\n%b"
    alignment = left
    word_wrap = yes
    stack_duplicates = true
    icon_position = left
    max_icon_size = 32

[urgency_low]
    background = "#1a1b26"
    foreground = "#a9b1d6"
    timeout = 5

[urgency_normal]
    background = "#1f2335"
    foreground = "#a9b1d6"
    timeout = 10

[urgency_critical]
    background = "#f7768e"
    foreground = "#1a1b26"
    timeout = 0
DUNST

# xinitrc
cat > "$UH/.xinitrc" <<'XINITRC'
#!/bin/sh
export XDG_SESSION_TYPE=x11
export XDG_CURRENT_DESKTOP=bspwm
export GTK_THEME=Adwaita:dark
export QT_QPA_PLATFORMTHEME=qt5ct
export _JAVA_AWT_WM_NONREPARENTING=1
exec bspwm
XINITRC
chmod +x "$UH/.xinitrc"

cat > "$UH/.xprofile" <<'XPROFILE'
export XDG_SESSION_TYPE=x11
export XDG_CURRENT_DESKTOP=bspwm
export GTK_THEME=Adwaita:dark
export QT_QPA_PLATFORMTHEME=qt5ct
export _JAVA_AWT_WM_NONREPARENTING=1
XPROFILE

arch-chroot /mnt chown -R "${USERNAME}:${USERNAME}" \
    "/home/${USERNAME}/.config" \
    "/home/${USERNAME}/.xinitrc" \
    "/home/${USERNAME}/.xprofile" \
    "/home/${USERNAME}/.local" \
    "/home/${USERNAME}/Pictures"

info "Конфигурация WM создана."

# ══════════════════════════════════════════════════════════════
#  18. YAY + AUR ПАКЕТЫ
# ══════════════════════════════════════════════════════════════
if [[ -n "$AUR_PKGS" ]]; then
    echo -e "\n${BOLD}=== Установка AUR-пакетов ===${NC}"
    info "AUR: $AUR_PKGS"

    arch-chroot /mnt /bin/bash -e <<CHROOT_YAY
set -e
if ! command -v yay &>/dev/null; then
    cd /tmp
    sudo -u ${USERNAME} git clone https://aur.archlinux.org/yay.git
    cd yay
    sudo -u ${USERNAME} makepkg -si --noconfirm
fi
sudo -u ${USERNAME} yay -S --noconfirm --needed ${AUR_PKGS}
if echo "${AUR_KERNEL_PKGS}" | grep -qw "linux"; then
    grub-mkconfig -o /boot/grub/grub.cfg
fi
CHROOT_YAY

    info "AUR-пакеты установлены."
else
    info "AUR-пакеты не требуются."
fi

# ══════════════════════════════════════════════════════════════
#  19. FLATPAK
# ══════════════════════════════════════════════════════════════
if [[ ${#FLATPAK_PKGS[@]} -gt 0 ]]; then
    echo -e "\n${BOLD}=== Установка Flatpak ===${NC}"

    arch-chroot /mnt /bin/bash -e <<CHROOT_FLATPAK
set -e
pacman -S --noconfirm --needed flatpak
flatpak remote-add --if-not-exists flathub https://flathub.org/repo/flathub.flatpakrepo
for app in ${FLATPAK_PKGS[*]}; do
    echo "Установка: \$app"
    flatpak install -y flathub "\$app" || echo "Не удалось установить: \$app"
done
CHROOT_FLATPAK

    info "Flatpak: ${#FLATPAK_PKGS[@]} приложений установлено."
else
    info "Flatpak приложения не выбраны."
fi

# ══════════════════════════════════════════════════════════════
#  20. СВОДКА
# ══════════════════════════════════════════════════════════════
echo ""
echo -e "${CYAN}╔═══════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║              УСТАНОВКА ЗАВЕРШЕНА                           ║${NC}"
echo -e "${CYAN}╠═══════════════════════════════════════════════════════════╣${NC}"
echo -e "${CYAN}║ Хост:         ${HOSTNAME}${NC}"
echo -e "${CYAN}║ Пользователь: ${USERNAME}${NC}"
echo -e "${CYAN}║ Ядра:         ${REPO_KERNEL_PKGS} ${AUR_KERNEL_PKGS}${NC}"
echo -e "${CYAN}║ ФС:           ${ROOT_FS}${NC}"
echo -e "${CYAN}║ Аудио:        ${AUDIO_SERVER}${NC}"
echo -e "${CYAN}║ Видео:        ${VIDEO_DRIVER}${NVIDIA_VARIANT:+ ($NVIDIA_VARIANT)}${NC}"
echo -e "${CYAN}║ Терминал:     ${DEFAULT_TERM}${NC}"
echo -e "${CYAN}║ Файловик:     ${DEFAULT_FM}${NC}"
echo -e "${CYAN}║ Браузер:      ${DEFAULT_BROWSER}${NC}"
echo -e "${CYAN}║ Редактор:     ${DEFAULT_EDITOR}${NC}"
echo -e "${CYAN}║ Flatpak:      ${#FLATPAK_PKGS[@]} приложений${NC}"
echo -e "${CYAN}║ AUR пакетов:  $(echo "$AUR_PKGS" | wc -w)${NC}"
echo -e "${CYAN}║ Репо пакетов: $(echo "$REPO_PKGS" | wc -w)${NC}"
echo -e "${CYAN}║ Раскладка:    ${KBD_OPTS}${NC}"
echo -e "${CYAN}╚═══════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${YELLOW}После перезагрузки:${NC}"
echo "  1. Войдите как: ${USERNAME}"
echo "  2. Запуск X:    startx"
echo "  3. Меню:        Super+D (rofi)"
echo "  4. Терминал:    Super+Enter ($DEFAULT_TERM)"
echo "  5. Файловик:    Super+E ($DEFAULT_FM)"
echo "  6. Браузер:     Super+W ($DEFAULT_BROWSER)"
echo "  7. Редактор:    Super+N ($DEFAULT_EDITOR)"
echo ""
read -p "Перезагрузить сейчас? (y/N): " REBOOT
if [[ "$REBOOT" =~ ^[Yy]$ ]]; then
    umount -R /mnt
    reboot
fi
