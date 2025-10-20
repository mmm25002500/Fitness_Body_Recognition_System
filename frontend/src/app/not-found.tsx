import Link from "next/link";
import type { Metadata } from "next";
import SEO from "@/config/SEO.json";

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
    <main className="min-h-screen dark:bg-neutral-900 flex items-center justify-center">
      <div className="container mx-auto px-4 py-8">
        <div className="text-center space-y-6">
          <h1 className="text-9xl font-bold text-bityo">404</h1>
          <h2 className="text-4xl font-bold text-white">找不到頁面</h2>
          <p className="text-gray-400 text-lg max-w-md mx-auto">
            抱歉，您訪問的頁面不存在或已被移除。
          </p>

          <div className="pt-8">
            <Link
              href="/"
              className="inline-block px-8 py-4 bg-bityo hover:bg-bityo/80 text-white font-semibold rounded-lg transition-colors"
            >
              返回首頁
            </Link>
          </div>

          <div className="pt-12 text-gray-500 text-sm">
            <p>如果您認為這是一個錯誤，請聯繫我們。</p>
          </div>
        </div>
      </div>
    </main>
  );
}
