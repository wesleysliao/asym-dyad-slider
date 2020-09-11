{ pkgs ? import <nixpkgs> {} }:

let
  customPython = pkgs.python38.buildEnv.override {
    extraLibs = with pkgs.python38Packages; [ gym numpy scipy ];
  };
in

pkgs.mkShell {
  buildInputs = [ customPython ];
}
