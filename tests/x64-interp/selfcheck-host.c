// SPDX-License-Identifier: GPL-3.0-or-later
// Madeira Converter Exception: see LICENSE-EXCEPTION.md
//
// Runs the app's No JIT engine check on the build host (any architecture):
// the same code the iPhone runs from Settings, built for this machine.

#include "NoJITSelfCheck.h"

#include <stdio.h>

int main(void)
{
    MadeiraNoJITReport report;
    char text[16384];
    bool ok = madeira_nojit_self_check(&report, text, sizeof text);
    fputs(text, stdout);
    return ok ? 0 : 1;
}
