{ pkgs ? import (fetchTarball "https://github.com/NixOS/nixpkgs/tarball/nixos-24.05") {} }:

let
  python = pkgs.python312;
in
pkgs.mkShellNoCC {
  packages = with pkgs; [
    (python.withPackages (ps: with ps; [
      numpy
      matplotlib
    ]))
  ];
}
