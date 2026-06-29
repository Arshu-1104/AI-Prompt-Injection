import { copyFileSync, cpSync, mkdirSync } from "node:fs";

mkdirSync("dist", { recursive: true });
copyFileSync("manifest.json", "dist/manifest.json");
copyFileSync("options.html", "dist/options.html");
copyFileSync("popup.html", "dist/popup.html");
cpSync("icons", "dist/icons", { recursive: true });
