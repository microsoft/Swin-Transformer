#!/usr/env bash

cat $1 | rg '190/196.*Acc@1' | nl | rg ' 1\t'
cat $1 | rg '190/196.*Acc@1' | nl | rg ' 30\t'
cat $1 | rg '190/196.*Acc@1' | nl | rg ' 60\t'
cat $1 | rg '190/196.*Acc@1' | nl | rg ' 90\t'
