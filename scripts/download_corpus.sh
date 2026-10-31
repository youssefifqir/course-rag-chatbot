#!/usr/bin/env bash
# Download the public course corpus into data/raw/.
# All documents are freely available public course notes/books (see README credits).
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data/raw/course

fetch () {
  local url="$1" out="data/raw/course/$2" min_bytes="$3"
  if [ -f "$out" ] && [ "$(wc -c < "$out")" -ge "$min_bytes" ]; then
    echo "[ok] $2 already present"
    return
  fi
  echo "[get] $url"
  # NB: ciml.info serves a certificate under a different hostname, hence -k for
  # that host only. Content is the author-released PDF from ciml.info.
  if [[ "$url" == "https://ciml.info"* ]]; then
    curl -fkL --retry 2 -o "$out" "$url"
  else
    curl -fL --retry 2 -o "$out" "$url"
  fi
  if ! head -c 5 "$out" | grep -q "%PDF"; then
    echo "$out is not a PDF - aborting"; exit 1
  fi
  echo "[ok] $2 ($(wc -c < "$out") bytes)"
}

# "A Course in Machine Learning" (CIML) v0.99 - Hal Daume III (free online edition)
fetch https://ciml.info/dl/v0_99/ciml-v0_99-all.pdf ciml_machine_learning.pdf 1500000

# "Think Python" 2nd ed. - Allen B. Downey (free, CC BY-NC)
fetch https://greenteapress.com/thinkpython2/thinkpython2.pdf think_python.pdf 900000

# "Speech and Language Processing" 3rd ed. draft chapters - Jurf (free draft for coursework)
fetch https://web.stanford.edu/~jurafsky/slp3/7.pdf  slp3_ch7_transformers.pdf 600000
fetch https://web.stanford.edu/~jurafsky/slp3/9.pdf  slp3_ch9_masked_lm.pdf   300000
fetch https://web.stanford.edu/~jurafsky/slp3/11.pdf slp3_ch11_ir_rag.pdf    300000

echo "done. Next: python -m src.ingest"
