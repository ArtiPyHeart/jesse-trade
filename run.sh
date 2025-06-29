#!/bin/bash

# 检查是否为 Linux 系统
if [[ "$(uname)" == "Linux" ]]; then
    systemctl restart pgbouncer
fi

jesse run