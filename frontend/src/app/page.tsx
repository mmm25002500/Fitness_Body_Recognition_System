
import type { Metadata } from "next";
import SEO from "@/config/SEO.json";
import HomeClient from '@/components/HomeClient';

export const metadata: Metadata = {
  title: SEO.Index.title,
  description: SEO.Index.description,
  keywords: SEO.Index.keyword,
  authors: [{ name: SEO.Index.author }],
  openGraph: {
    title: SEO.Index.title,
    description: SEO.Index.description,
    type: SEO.Index.type as "website",
    images: [
      {
        url: SEO.Index.image,
        alt: SEO.Index.title,
      },
    ],
  },
  twitter: {
    card: "summary_large_image",
    title: SEO.Index.title,
    description: SEO.Index.description,
    images: [SEO.Index.image],
  },
};


const Home = () => {
  return (
    <main className="min-h-screen dark:bg-neutral-900">
      <div className="container mx-auto px-4 py-8">
        <HomeClient />
      </div>
    </main>
  );
}

export default Home;