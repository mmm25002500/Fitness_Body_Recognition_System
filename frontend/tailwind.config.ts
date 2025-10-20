// tailwind.config.ts
import type { Config } from "tailwindcss";
import lineClamp from "@tailwindcss/line-clamp";

const config: Config = {
	darkMode: "class",
	content: [
		"./app/**/*.{js,ts,jsx,tsx,mdx}",
		"./components/**/*.{js,ts,jsx,tsx,mdx}",
		"./src/**/*.{js,ts,jsx,tsx,mdx}",
	],
	theme: {
		extend: {
			keyframes: {
				fadeIn: {
					"0%": { opacity: "0" },
					"100%": { opacity: "1" },
				},
				fadeOut: {
					"0%": { opacity: "1" },
					"100%": { opacity: "0" },
				},
			},
			animation: {
				fadeIn: "fadeIn 0.5s ease-in-out",
				fadeOut: "fadeOut 0.5s ease-in-out",
			},
			backgroundImage: {
				"gradient-radial": "radial-gradient(var(--tw-gradient-stops))",
				"gradient-conic":
					"conic-gradient(from 180deg at 50% 50%, var(--tw-gradient-stops))",
			},
			fontSize: {
				h1: ["5rem", "7.5rem"],
				h2: ["4rem", "6rem"],
				h3: ["3rem", "4.5rem"],
				h4: ["2.5rem", "3.75rem"],
				h5: ["2rem", "3rem"],
				h6: ["1.75rem", "2.625rem"],
			},
			colors: {
				"web-green": "#4ECE80",
				bityo: "#17FFAD",
				body: "#788293",
				"body-dark": "rgb(149,156,177)",
				"txt-dark": "rgba(41,44,53)",
				primary: {
					"black-100": "#000000",
					"black-200": "#373737",
					"black-300": "#1C1C1C",
					"black-400": "#CECECE",
					"black-500": "#EEF0F6",
				},
				neutral: {
					white: "#FFFFFF",
					black: "#1A1A1A",
					"900": "#3D3D3D",
					"800": "#545454",
					"700": "#6B6B6B",
					"600": "#828282",
					"500": "#999999",
					"400": "#B0B0B0",
					"300": "#C7C7C7",
					"200": "#DEDEDE",
					"100": "#F5F5F5 ",
					"tone-700": "#F2F3F5",
				},
				wireframe: {
					700: "#BCBACD",
				},
				func: {
					error: "#D82027",
					info: "#208BD8",
					warning: "#EDE405",
					success: "#20D880",
				},
			},
		},
	},
	plugins: [lineClamp],
};

export default config;
