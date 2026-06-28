#!/usr/bin/env bash
#
# deepseek-claude.sh — point Claude Code at DeepSeek V4 (Anthropic-compatible API)
#
# Usage:
#   source ./deepseek-claude.sh on      # route Claude Code -> DeepSeek (this shell only)
#   source ./deepseek-claude.sh off     # revert to normal Anthropic / Claude
#   source ./deepseek-claude.sh status  # show what's currently active
#
# NOTE: must be *sourced*, not executed, so the env vars apply to your shell.
#   Good:  source ./deepseek-claude.sh on
#   Bad:   ./deepseek-claude.sh on   (changes vanish when the script exits)
#
# Your DeepSeek API key is read from ~/.deepseek_key (preferred) or the
# DEEPSEEK_API_KEY env var. The key is never written into this file.

# ---- config -----------------------------------------------------------------
DEEPSEEK_BASE_URL="https://api.deepseek.com/anthropic"
DEEPSEEK_MAIN_MODEL="deepseek-v4-pro"     # maps to Opus/Sonnet roles
DEEPSEEK_FAST_MODEL="deepseek-v4-flash"   # maps to Haiku role
KEY_FILE="$HOME/.deepseek_key"
# -----------------------------------------------------------------------------

_ds_load_key() {
    if [ -n "$DEEPSEEK_API_KEY" ]; then
        printf '%s' "$DEEPSEEK_API_KEY"
    elif [ -f "$KEY_FILE" ]; then
        # strip whitespace/newlines
        tr -d '[:space:]' < "$KEY_FILE"
    else
        return 1
    fi
}

_ds_on() {
    local key
    key="$(_ds_load_key)" || {
        echo "✗ No DeepSeek key found." >&2
        echo "  Put it in $KEY_FILE  (chmod 600), e.g.:" >&2
        echo "    echo 'sk-your-deepseek-key' > $KEY_FILE && chmod 600 $KEY_FILE" >&2
        echo "  ...or:  export DEEPSEEK_API_KEY=sk-..." >&2
        return 1
    }

    # Stash any existing real-Anthropic values so 'off' can restore them.
    export _DS_SAVED_BASE_URL="${ANTHROPIC_BASE_URL-__unset__}"
    export _DS_SAVED_AUTH_TOKEN="${ANTHROPIC_AUTH_TOKEN-__unset__}"
    export _DS_SAVED_MODEL="${ANTHROPIC_MODEL-__unset__}"
    export _DS_SAVED_FAST="${ANTHROPIC_SMALL_FAST_MODEL-__unset__}"

    export ANTHROPIC_BASE_URL="$DEEPSEEK_BASE_URL"
    export ANTHROPIC_AUTH_TOKEN="$key"
    export ANTHROPIC_MODEL="$DEEPSEEK_MAIN_MODEL"
    export ANTHROPIC_SMALL_FAST_MODEL="$DEEPSEEK_FAST_MODEL"
    export CLAUDE_CODE_EFFORT_LEVEL="max"
    # Avoid clashes: an ANTHROPIC_API_KEY would override AUTH_TOKEN.
    unset ANTHROPIC_API_KEY

    echo "✓ Claude Code -> DeepSeek ($DEEPSEEK_MAIN_MODEL / $DEEPSEEK_FAST_MODEL)"
    echo "  Endpoint: $DEEPSEEK_BASE_URL"
    echo "  Start a NEW 'claude' session in this shell for it to take effect."
}

_ds_restore() {
    local saved="$1" var="$2"
    if [ "$saved" = "__unset__" ] || [ -z "$saved" ]; then
        unset "$var"
    else
        export "$var=$saved"
    fi
}

_ds_off() {
    _ds_restore "${_DS_SAVED_BASE_URL-__unset__}"   ANTHROPIC_BASE_URL
    _ds_restore "${_DS_SAVED_AUTH_TOKEN-__unset__}" ANTHROPIC_AUTH_TOKEN
    _ds_restore "${_DS_SAVED_MODEL-__unset__}"      ANTHROPIC_MODEL
    _ds_restore "${_DS_SAVED_FAST-__unset__}"       ANTHROPIC_SMALL_FAST_MODEL
    unset CLAUDE_CODE_EFFORT_LEVEL
    unset _DS_SAVED_BASE_URL _DS_SAVED_AUTH_TOKEN _DS_SAVED_MODEL _DS_SAVED_FAST
    echo "✓ Reverted to normal Anthropic / Claude. Start a new 'claude' session."
}

_ds_status() {
    if [ "${ANTHROPIC_BASE_URL-}" = "$DEEPSEEK_BASE_URL" ]; then
        echo "ACTIVE: DeepSeek"
        echo "  base : $ANTHROPIC_BASE_URL"
        echo "  model: ${ANTHROPIC_MODEL-?}  fast: ${ANTHROPIC_SMALL_FAST_MODEL-?}"
        echo "  token: ${ANTHROPIC_AUTH_TOKEN:0:6}... (hidden)"
    else
        echo "ACTIVE: normal Anthropic / Claude (DeepSeek off)"
    fi
}

case "${1:-status}" in
    on)     _ds_on ;;
    off)    _ds_off ;;
    status) _ds_status ;;
    *)      echo "Usage: source ${BASH_SOURCE[0]:-$0} {on|off|status}" >&2 ;;
esac
