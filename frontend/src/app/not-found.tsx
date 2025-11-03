import Link from "next/link";
import type { Metadata } from "next";
import SEO from "@/config/SEO.json";
import Footer from "@/components/Footer/Footer";
import Navbar from "@/components/Layout/Navbar";

export const metadata: Metadata = {
  title: SEO.NotFound.title,
  description: SEO.NotFound.description,
  keywords: SEO.NotFound.keyword,
  authors: [{ name: SEO.NotFound.author }],
  openGraph: {
    title: SEO.NotFound.title,
    description: SEO.NotFound.description,
    type: SEO.NotFound.type as "website",
    images: [
      {
        url: SEO.NotFound.image,
        alt: SEO.NotFound.title,
      },
    ],
  },
  twitter: {
    card: "summary_large_image",
    title: SEO.NotFound.title,
    description: SEO.NotFound.description,
    images: [SEO.NotFound.image],
  },
};

export default function NotFound() {
  return (
    <main className="min-h-screen bg-gray-950 flex flex-col">
      <Navbar />
      <div className="flex flex-col flex-1 items-center justify-center">
        <div className="text-center space-y-6">
          <h1 className="text-9xl font-bold text-func-error">404</h1>
          <h2 className="text-4xl font-bold text-func-error">找不到頁面</h2>
          <div className="pt-8">
            <Link
              href="/"
              className="inline-block px-8 py-3 bg-neutral-black hover:bg-neutral-800 text-neutral-white rounded-lg transition-colors"
            >
              返回首頁
            </Link>
          </div>
        </div>
      </div>
      <Footer />
    </main>
  );
}
