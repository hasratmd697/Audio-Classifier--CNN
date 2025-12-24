"use client";

import dynamic from "next/dynamic";

const HomePageContent = dynamic(
  () => import("../components/HomePageContent"),
  { ssr: false }
);

export default function HomePage() {
  return <HomePageContent />;
}
