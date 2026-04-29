import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "DxMap",
  description: "ICD-10-CM and CPT code suggestions with span attribution",
  icons: {
    icon: "/DxMap.png",
    apple: "/DxMap.png",
  },
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
